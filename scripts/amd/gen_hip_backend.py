import argparse
from collections import OrderedDict, defaultdict
import copy
from enum import Enum
from fnmatch import translate
from genericpath import isfile
import json
from logging import debug, error, info, warning
from inspect import stack
import logging
from multiprocessing import Value
import os
from pydoc import cli
import shutil
import stat
from telnetlib import EL
import time
import traceback
import chardet
import tree_sitter
from pathlib import Path
from shutil import copytree
from distutils import dir_util
from typing import List, Dict, Type, Union
from git import Repo
from git.exc import GitCommandError
import os
import multiprocessing
from tqdm import tqdm


def get_rewrite_function():
    cur_stack = stack()
    rewrite_func_names = [f.__name__ for f in REWRITE_FUNCTIONS]
    for frame_info in cur_stack:
        fun_name = frame_info[3]
        if fun_name in rewrite_func_names:
            return fun_name  # return the first language rewrite function it finds


def rewrite_cpp_defines(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["preproc_def", "preproc_ifdef"]:
        preproc_def = node
        for child in preproc_def.children:
            if child.type == "identifier":
                rewrite_loop(child, code, edits, rewrite_map["defines"])
            elif child.type == "preproc_arg":
                rewrite_loop(child, code, edits, rewrite_map["preproc_args"])
    elif node.type in ["preproc_function_def"]:
        preproc_def = node
        for child in preproc_def.children:
            if child.type == "identifier":
                rewrite_loop(child, code, edits, rewrite_map["defines"])
                rewrite_loop(child, code, edits, rewrite_map["functions"])
            elif child.type == "preproc_arg":
                rewrite_loop(child, code, edits, rewrite_map["preproc_args"])
                rewrite_loop(child, code, edits, rewrite_map["functions"])


def rewrite_cpp_includes(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type == "preproc_include":
        if node.children[1].type == "string_literal":
            string_literal = node.children[1]
            string_content = string_literal.children[1]
            string_content_str = get_node_str(string_content, code)

            for src, tgt, strict in rewrite_map["includes"]:
                add_edit(node, code, edits, (src, tgt, strict), strict=strict)


def rewrite_cpp_use_rocm(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["preproc_ifdef", "preproc_if", "ERROR"]:
        preproc_type = node.children[0]
        preproc_type_str = get_node_str(preproc_type, code)
        if len(node.children) > 1:
            identifer = node.children[1]
            identifer_str = get_node_str(identifer, code)
            if "USE_ROCM" in identifer_str:
                if preproc_type_str in ["#if", "#ifdef"]:
                    edits.append(
                        (preproc_type.start_byte, preproc_type.end_byte, "#if")
                    )
                    edits.append((identifer.start_byte, identifer.end_byte, "1"))
                elif preproc_type_str in ["#ifndef"]:
                    edits.append(
                        (preproc_type.start_byte, preproc_type.end_byte, "#if")
                    )
                    edits.append((identifer.start_byte, identifer.end_byte, "0"))
        else:
            pass


def rewrite_cpp_use_rocm_bad(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    # NOTE: the parse has issues with some USE_ROCM

    # TODO: this is brittle. white space is an issue
    node_code = code[node.start_byte : node.end_byte]
    new_node_code = (
        node_code.replace("#ifdef USE_ROCM", "#if 1")
        .replace("#ifndef USE_ROCM", "#if 0")
        .replace("#if USE_ROCM", "#if 1")
    )
    edits.append((node.start_byte, node.end_byte, new_node_code))


def traverse_node_and_exec_fn(node: tree_sitter.Node, code: str, fn, *args, **kwargs):
    stack = [(node, 0)]
    while stack:
        node, indent = stack.pop()
        if is_in_range(node):
            fn(node, code, *args, **kwargs)

        stack.extend(
            (field_child, indent + 2) for field_child in reversed(node.children)
        )


def rewrite_cpp_namespaces(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    def translate_namespaces(node, code, edits):
        if node.type in ["namespace_identifier", "identifier"]:
            for src, tgt, strict in rewrite_map["namespaces"]:
                add_edit(node, code, edits, (src, tgt, strict), strict=strict)

    def is_valid_namespace(namespace_list):
        relevant_namespaces = [
            "mlir",
            "triton",
            "gpu",
            "TritonGPU",
            "nvgpu",
            "TritonNvidiaGPUDialect",
        ]
        present_namespaces = defaultdict(lambda: False)
        for identifier in namespace_list:
            for n in relevant_namespaces:
                if n in identifier:
                    present_namespaces[n] = True

        if present_namespaces["nvgpu"]:
            return True  # skip nvgpu stuff
        elif (
            present_namespaces["mlir"]
            and present_namespaces["triton"]
            and present_namespaces["gpu"]
        ):
            return True
        elif present_namespaces["triton"] and present_namespaces["gpu"]:
            return True
        elif present_namespaces["mlir"] and present_namespaces["triton"]:
            return True
        elif present_namespaces["triton"]:
            return True
        elif present_namespaces["TritonGPU"]:
            return True
        elif present_namespaces["TritonNvidiaGPUDialect"]:
            return True
        else:
            debug(f"{namespace_list} namespace should not be rewritten")
            return False

    def traverse_node_and_rewrite_namespaces(node: tree_sitter.Node, code: str):
        stack = [(node, 0)]
        while stack:
            node, indent = stack.pop()
            if node.type in ["namespace_identifier", "identifier"]:
                for src, tgt, strict in rewrite_map["namespaces"]:
                    add_edit(node, code, edits, (src, tgt, strict), strict=strict)
            elif node.type in ["qualified_identifier", "nested_namespace_specifier"]:
                # allow traversal to continue
                stack.extend(
                    (field_child, indent + 2) for field_child in reversed(node.children)
                )
            else:
                # don't traverse
                pass

    if node.type in ["qualified_identifier"] and node.parent.type in [
        "type_descriptor",
        "declaration",
        "parameter_declaration",
        "using_declaration",
        "field_declaration",
        "function_definition",
        "call_expression",
        "function_declarator",
        "argument_list",
        "binary_expression",
        "friend_declaration",
        "field_expression",
        "for_range_loop",
        "base_class_clause",
        "pointer_expression",
        "dependent_type",
        "optional_parameter_declaration",
        "return_statement",
        "assignment_expression",
        "conditional_expression",
        "init_declarator",
    ]:
        qualified_identifier = node
        qualified_identifier_str = get_node_str(node, code)

        # if "tt::nvidia_gpu:" in qualified_identifier_str:
        #     add_edit(node, code, edits, ("tt::nvidia_gpu:","triton::nvidia_gpu:", False), force=True)
        #     return

        if shouldSkip(qualified_identifier, code):
            return

        # collect all the namespace identifiers
        list_of_namespaces = []
        traverse_node_and_exec_fn(
            qualified_identifier,
            code,
            lambda node, code: list_of_namespaces.append(get_node_str(node, code))
            if node.type in ["namespace_identifier", "identifier"]
            else None,
        )
        # if valid namespace modify
        if is_valid_namespace(list_of_namespaces):
            traverse_node_and_rewrite_namespaces(node, code)
    elif node.type in ["namespace_definition"]:
        namespace_identifier = node.children[1]
        translate_namespaces(namespace_identifier, code, edits)
    elif node.type in ["nested_namespace_specifier"] and node.parent.type in [
        "namespace_alias_definition", "namespace_definition"
    ]:
        nested_namespace_specifier = node

        if shouldSkip(nested_namespace_specifier, code):
            return

        # collect all the namespace identifiers
        list_of_namespaces = []
        traverse_node_and_exec_fn(
            nested_namespace_specifier,
            code,
            lambda node, code: list_of_namespaces.append(get_node_str(node, code))
            if node.type in ["namespace_identifier", "nested_namespace_specifier"]
            else None,
        )
        # if valid namespace modify
        if is_valid_namespace(list_of_namespaces):
            traverse_node_and_rewrite_namespaces(nested_namespace_specifier, code)


def rewrite_cpp_types(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["type_identifier"]:
        rewrite_loop(node, code, edits, rewrite_map["types"])
    elif node.type in ["init_declarator"]:
        identifier = node.children[0]
        rewrite_loop(identifier, code, edits, rewrite_map["types"])
    elif node.type == "identifier" and node.parent.type in [
        "binary_expression",
        "qualified_identifier",
    ]:
        identifier = node
        rewrite_loop(identifier, code, edits, rewrite_map["types"])


REWRITTEN_LIST = [
    "gpu_rocm",
    "triton_rocm",
    "TritonROCMDialect",
    "TritonGPUROCM",
    "librocm",
    "ROCM",
]
# SKIP_LIST = ["TMAMetadataTy", "TMAInfo", "createTritonGPURewriteTensorPointerPass", ]
SKIP_LIST = []


def shouldSkip(node, code):
    node_str = get_node_str(node, code)
    for skip_word in SKIP_LIST:
        if skip_word in node_str:
            debug(
                f'skipped "{get_node_str_pretty(node, code)}" contains "{skip_word}" at line {node.start_point[0]}. Skipping rewrite in {get_rewrite_function()}'
            )
            return True

    return False


def is_match(node, code, rewrite):
    src, target, strict = rewrite
    node_str = get_node_str(node, code)
    # check if src in node_str
    if strict:
        if not (src == node_str):
            debug(
                f'cannot rewrite "{get_node_str_pretty(node, code)}" is not equal to "{src}" at line {node.start_point[0]}. It was called from {get_rewrite_function()}'
            )
            return False
    else:
        if not (src in node_str):
            debug(
                f'cannot rewrite "{get_node_str_pretty(node, code)}" missing "{src}" at line {node.start_point[0]}. It was called from {get_rewrite_function()}'
            )
            return False

    return True


def can_rewrite(node: tree_sitter.Node, code: str, rewrite: tuple):
    node_str = get_node_str(node, code)

    if not is_match(node, code, rewrite):
        return False

    if shouldSkip(node, code):
        return False

    for rewrite_word in REWRITTEN_LIST:
        if rewrite_word in node_str:
            target = rewrite[1]
            if rewrite_word in target:
                debug(
                    f'already rewrote "{get_node_str_pretty(node, code)}" at {node.start_point[0]}. It contains "{rewrite_word}". Didnot apply {rewrite} rewrite in {get_rewrite_function()}'
                )
                return False

    return True


def traverse_node_and_rewrite_identifiers(
    node: tree_sitter.Node, code: str, edits, rewrites
):
    stack = [(node, 0)]
    while stack:
        node, indent = stack.pop()
        if node.type in ["identifier", "namespace_identifier"]:
            for src, tgt, strict in rewrites:
                add_edit(node, code, edits, (src, tgt, strict), strict=strict)
        elif node.type in ["qualified_identifier"]:
            # allow traversal to continue
            stack.extend(
                (field_child, indent + 2) for field_child in reversed(node.children)
            )
        else:
            # don't traverse
            pass


def comment_out(code: str):
    return "/* " + code + " */"


# COMMENT_OUT_FUNCTION_LIST = ["init_triton_runtime"]
COMMENT_OUT_FUNCTION_LIST = []


def rewrite_cpp_functions(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    def translate_functions(node, code, edits):
        for src, tgt, strict in rewrite_map["functions"]:
            add_edit(node, code, edits, (src, tgt, strict), strict=strict)

    def translate_types(node, code, edits):
        for src, tgt, strict in rewrite_map["types"]:
            add_edit(node, code, edits, (src, tgt, strict), strict=strict)

    if node.type in ["function_declarator"]:
        if node.children[0].type == "identifier":
            identifier = node.children[0]
            translate_functions(identifier, code, edits)
        elif node.children[0].type == "qualified_identifier":
            qualified_identifier = node.children[0]
            traverse_node_and_rewrite_identifiers(
                qualified_identifier, code, edits, rewrite_map["functions"]
            )

        # rewrite the params
        if node.children[1].type == "parameter_list":
            parameter_list = node.children[1]
            for field_child in parameter_list.children:
                if field_child.type == "parameter_declaration":
                    translate_types(field_child, code, edits)
    elif node.type in ["field_initializer"]:  # they look like functions!
        if node.children[0].type == "field_identifier":
            field_identifier = node.children[0]
            translate_functions(field_identifier, code, edits)
        elif node.children[0].type == "template_method":
            template_method = node.children[0]
            if template_method.children[0].type == "field_identifier":
                field_identifier = template_method.children[0]
                translate_functions(field_identifier, code, edits)

        # rewrite the args
        if node.children[1].type == "argument_list":
            argument_list = node.children[1]
            traverse_node_and_exec_fn(
                argument_list,
                code,
                lambda node, code: translate_functions(node, code, edits)
                if node.type in ["identifier"]
                else None,
            )
    elif node.type in ["call_expression"]:
        call_expression = node
        call_expression_str = get_node_str(node, code)

        if call_expression.children[0].type in [
            "identifier",
            "qualified_identifier",
        ]:  # regular function
            identifier = call_expression.children[0]
            identifier_str = get_node_str(identifier, code)

            for f in COMMENT_OUT_FUNCTION_LIST:
                if f in identifier_str:
                    add_edit(
                        call_expression,
                        code,
                        edits,
                        (call_expression_str, comment_out(call_expression_str), False),
                        force=True,
                    )

            # rewrite function name
            translate_functions(identifier, code, edits)

            # remove args
            if identifier_str in ["mlir::triton::createRewriteTensorPointerPass"]:
                if node.children[1].type == "argument_list":
                    argument_list = node.children[1]
                    argument_list_str = get_node_str(argument_list, code)
                    if "isROCM" in argument_list_str:
                        add_edit(
                            argument_list,
                            code,
                            edits,
                            ("computeCapability, isROCM", "computeCapability"),
                        )

            # rewrite the args
            if node.children[1].type == "argument_list":
                argument_list = node.children[1]
                traverse_node_and_exec_fn(
                    argument_list,
                    code,
                    lambda node, code: translate_functions(node, code, edits)
                    if node.type in ["identifier"]
                    else None,
                )


def rewrite_cpp_functions_comment_out(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["call_expression"]:
        call_expression = node
        call_expression_str = get_node_str(node, code)
        if (
            call_expression.children[0].type in ["field_expression"]
            and call_expression.parent.type == "expression_statement"
        ):  # handle .def call expression in triton_rocm.cc
            # if "self.addPass(mlir::triton::createConvertNVGPUToLLVMPass())" == call_expression_str:
            #     add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str) , False), force=True)
            # if "translate_llvmir_to_ptx" in call_expression_str:
            #     add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str), False), force=True)

            # check if top level call expression
            # if call_expression.parent and call_expression.parent.type == "expression_statement":
            #     if "py::class_<mlir::triton::nvidia_gpu::ClusterInfo>" in call_expression_str:
            #         add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str), False))
            #     elif "py::class_<mlir::triton::gpu::TMAInfo>" in call_expression_str:
            #         add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str), False), force=True)
            #     elif "py::bind_vector<std::vector<mlir::triton::gpu::TMAInfo>>" in call_expression_str:
            #         add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str), False), force=True)
            pass

        # if call_expression_str.startswith("self.getOrLoadDialect<mlir::"):
        #     add_edit(
        #         call_expression,
        #         code,
        #         edits,
        #         (call_expression_str, comment_out(call_expression_str), False),
        #         force=True,
        #     )
        # if call_expression_str.startswith("context.appendDialectRegistry"):
        #     add_edit(call_expression, code, edits, (call_expression_str, comment_out(call_expression_str)), force=True)
    elif node.type in ["function_definition"]:
        function_definition = node
        function_definition_str = get_node_str(function_definition, code)
        if node.children[1].type == "function_declarator":
            function_declarator = node.children[1]
            identifier = function_declarator
            identifier_str = get_node_str(identifier, code)
            for f in COMMENT_OUT_FUNCTION_LIST:
                if f in identifier_str:
                    add_edit(
                        function_definition,
                        code,
                        edits,
                        (
                            function_definition_str,
                            comment_out(function_definition_str),
                            False,
                        ),
                        force=True,
                    )
    elif node.type in ["template_method"]:
        template_method = node
        identifier = template_method.children[0]
        identifier_str = get_node_str(template_method, code)
        if False and "dialect_removal" in rewrite_map and "insert" in identifier_str:
            template_argument_list = template_method.children[1]
            call_expression = template_method.parent.parent
            call_expression_str = get_node_str(call_expression, code)
            add_edit(
                call_expression,
                code,
                edits,
                (call_expression_str, comment_out(call_expression_str)),
                strict=True,
                force=True,
            )

            if False:
                for i, child in enumerate(template_argument_list.children):
                    if child.type == "type_descriptor":
                        type_descriptor = child
                        type_descriptor_str = get_node_str(type_descriptor, code)
                        if type_descriptor_str not in [
                            "mlir::LLVM::LLVMDialect",
                            "mlir::triton::gpu_rocm::TritonGPUROCMDialect",
                        ]:  # ["mlir::arith::ArithDialect"]:
                            add_edit(
                                type_descriptor,
                                code,
                                edits,
                                (type_descriptor_str, ""),
                                strict=True,
                                force=True,
                            )
                            comma = template_argument_list.children[i + 1]
                            comma_str = get_node_str(comma, code)
                            if "," in comma_str:
                                add_edit(
                                    comma, code, edits, (comma_str, ""), strict=True
                                )


def rewrite_cpp_strings(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["string_content"]:
        for src, tgt, strict in rewrite_map["strings"]:
            add_edit(node, code, edits, (src, tgt, strict), strict=strict)


def rewrite_tablegen_preprocessor(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["preprocessor"]:
        rewrite_loop(node, code, edits, rewrite_map["preprocessor"])

    elif node.type in ["include"] and node.children:
        include_string = node.children[1]
        include_string_str = get_node_str(include_string, code)

        rewrite_loop(include_string, code, edits, rewrite_map["include"])


def rewrite_tablegen_definitions(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["def"] and node.children:
        defined_value = node.children[1]
        defined_value_str = get_node_str(defined_value, code)

        if "TritonGPU" in defined_value_str:
            add_edit(defined_value, code, edits, ("TritonGPU", "TritonGPUROCM", False))
        else:
            for src, tgt, strict in rewrite_map["defs"]:
                if src in defined_value_str:
                    add_edit(defined_value, code, edits, (src, tgt, strict))


def rewrite_tablegen_classes(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["class"] and node.children:
        class_identifier = node.children[1]
        class_identifier_str = get_node_str(class_identifier, code)

        if "TritonGPU" in class_identifier_str:
            add_edit(
                class_identifier, code, edits, ("TritonGPU", "TritonGPUROCM", False)
            )
        elif "TritonTypeDef" in class_identifier_str:
            add_edit(
                class_identifier,
                code,
                edits,
                ("TritonTypeDef", "TritonROCMTypeDef", False),
            )
        else:
            for src, tgt, strict in rewrite_map["classes"]:
                if src in class_identifier_str:
                    add_edit(node, code, edits, (src, tgt, strict))


def rewrite_tablegen_values(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["value"]:
        value = node
        if value.children:
            for child in value.children:
                if child.type == "identifier":
                    rewrite_loop(child, code, edits, rewrite_map["values"])


def rewrite_tablegen_parent_class(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["parent_class_list"]:
        parent_class_list = node
        for child in parent_class_list.children:
            child_str = get_node_str(node, code)
            if child.type == "identifier":
                rewrite_loop(child, code, edits, rewrite_map["parent_class_list"])
            elif child.type == "value":
                rewrite_loop(child, code, edits, rewrite_map["parent_class_list"])


def rewrite_tablegen_strings(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["string_string"]:
        rewrite_loop(node, code, edits, rewrite_map["strings"])
        rewrite_loop(
            node, code, edits, rewrite_map["values"]
        )  # values get refered to in strings sometimes


def rewrite_tablegen_code_strings(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    if node.type in ["code_string"]:  # maybe parse with CPP parser
        rewrite_loop(node, code, edits, rewrite_map["code_strings"])


def rewrite_cmake_arguments(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    def translate_cmake_arguments(argument_list, code, edits):
        for argument in argument_list.children:
            for src, tgt, strict in rewrite_map["argument"]:
                add_edit(argument, code, edits, (src, tgt, strict), strict=strict)

    if node.type in ["normal_command"]:
        identifier = node.children[0]
        argument_list = node.children[2]
        identifier_str = get_node_str(node, code)

        for command in [
            "mlir_tablegen",
            "add_public_tablegen_target",
            "add_mlir_library",
            "add_mlir_conversion_library",
            "add_mlir_dialect_library",
            "add_mlir_translation_library",
            "add_subdirectory"
        ]:
            if command in identifier_str:
                translate_cmake_arguments(argument_list, code, edits)

        if "add_mlir_doc" in identifier_str:
            # translate_cmake_arguments(argument_list, code, edits)
            for src, tgt, strict in rewrite_map["argument"]:
                add_edit(
                    argument_list.children[1],
                    code,
                    edits,
                    (src, tgt, strict),
                    strict=strict,
                )


def rewrite_loop(node: tree_sitter.Node, code: str, edits: list, map: list):
    for src, tgt, strict in map:
        add_edit(node, code, edits, (src, tgt, strict), strict=strict)


def rewrite_py_types(node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict):
    if node.type == "identifier" and node.parent.type in ["attribute", 
                                                          "generic_type", 
                                                          "class_definition", 
                                                          "call",
                                                          "type"
                                                          ]:
        identifier = node
        rewrite_loop(identifier, code, edits, rewrite_map["types"])

def rewrite_py_strings(node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict):
    if node.type == "string_content":
        string_content = node
        rewrite_loop(string_content, code, edits, rewrite_map["strings"])


def rewrite_py_functions(node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict):
    if node.type == "function_definition":
        function_definition = node
        identifier = function_definition.children[1]
        rewrite_loop(identifier, code, edits, rewrite_map["functions"])
    elif node.type in ["call"]:
        call = node
        for child in call.children:
            if child.type == "attribute":
                attribute = child
                for child in attribute.children:
                    if child.type == "identifier":
                        identifier = child
                        rewrite_loop(identifier, code, edits, rewrite_map["functions"])
            elif child.type == "identifier":
                identifier = child
                rewrite_loop(identifier, code, edits, rewrite_map["functions"])
            elif child.type == "argument_list":
                argument_list = child
                for arg in argument_list.children:
                    if arg.type == "attribute":
                        rewrite_loop(arg, code, edits, rewrite_map["types"])
                    elif arg.type == "identifier":
                        rewrite_loop(arg, code, edits, rewrite_map["types"])


def rewrite_py_import(
    node: tree_sitter.Node, code: str, edits: list, rewrite_map: dict
):
    def rewrite_dotted_name(dotted_name, code, edits, rewrite_map):
        if dotted_name.type == "dotted_name":
            for child in dotted_name.children:
                if child.type == "identifier":
                    identifier = child
                    rewrite_loop(identifier, code, edits, rewrite_map)
        else:
            raise TypeError(f"not a dotted name")
    
    def rewrite_aliased_import(aliased_import, code, edits, rewrite_map):
        if aliased_import.type in  [ "aliased_import"]:
            for child in aliased_import.children:
                if child.type == "dotted_name":
                    rewrite_dotted_name(child, code, edits, rewrite_map)
                elif child.type == "identifier":
                    rewrite_loop(child, code, edits, rewrite_map)
        else:
            raise TypeError(f"not a aliased_import")
        
        
    def rewrite_relative_import(relative_import, code, edits, rewrite_map):
        if relative_import.type == "relative_import":
            for child in relative_import.children:
                if child.type == "dotted_name":
                    rewrite_dotted_name(child, code, edits, rewrite_map)
                elif child.type == "import_prefix":
                    import_prefix = child
                    import_prefix_str = get_node_str(import_prefix, code)
                    if "libtriton" in import_from_statement_str:
                        add_edit(
                                import_prefix,
                                code,
                                edits,
                                (import_prefix_str, "....." + import_prefix_str, False),
                            )

    if node.type in ["import_statement"]:
        import_statement = node
        import_statement_str = get_node_str(import_statement, code)
        if "_C" in import_statement_str:
            for child in import_statement.children:
                if child.type == "aliased_import":
                    aliased_import = child
                    dotted_name = aliased_import.children[0]
                    after_shared_lib_dir = False
                    for child in dotted_name.children:
                        if child.type == "identifier":
                            identifier = child
                            if "_C" in get_node_str(identifier, code):
                                after_shared_lib_dir = True
                                continue
                            if after_shared_lib_dir:
                                rewrite_loop(child, code, edits, [
                                            (
                                                "libtriton",
                                                "librocm_backend_for_triton",
                                                True,
                                            ),
                                            ("triton", "triton_rocm", True),
                                        ])
        else:
            # check if it is an aliased_import
            is_aliased = False
            for child in import_statement.children:
                if child.type == "aliased_import":
                    is_aliased = True
                    rewrite_aliased_import(child, code, edits, rewrite_map["imports"])

            if not is_aliased:
                for child in import_statement.children:
                    if child.type == "dotted_name":
                        rewrite_dotted_name(child, code, edits, rewrite_map["imports"])


    elif node.type in ["import_from_statement"]:
        import_from_statement = node
        import_from_statement_str = get_node_str(import_from_statement, code)
        for i, child in enumerate(import_from_statement.children):
            if child.type == "relative_import":
                rewrite_relative_import(child, code, edits, rewrite_map["relative_imports"])
            elif child.type == "dotted_name":
                dotted_name = child
                
                for child in dotted_name.children:
                    if child.type == "identifier":
                        identifier = child
                        identifier_str = get_node_str(identifier, code)
                        if identifier_str == "get_backend":
                            add_edit(
                                import_from_statement,
                                code,
                                edits,
                                (import_from_statement_str, "# " + import_from_statement_str, False),
                            )
                        else:   
                            rewrite_loop(identifier, code, edits, rewrite_map["functions"])
                            rewrite_loop(identifier, code, edits, rewrite_map["types"])
        


# NOTE: order matters. Write now. We are going specific to broad
PYTHON_REWRITE_FUNCTIONS = [rewrite_py_import, rewrite_py_types, rewrite_py_functions, rewrite_py_strings]

CPP_REWRITE_FUNCTIONS = [
    rewrite_cpp_defines,
    rewrite_cpp_includes,
    rewrite_cpp_namespaces,
    rewrite_cpp_types,
    rewrite_cpp_functions,
    rewrite_cpp_strings,
    rewrite_cpp_use_rocm,
    rewrite_cpp_use_rocm_bad,
    rewrite_cpp_functions_comment_out,
]

CMAKE_REWRITE_FUNCTIONS = [rewrite_cmake_arguments]

TABLEGEN_REWRITE_FUNCTIONS = [
    rewrite_tablegen_preprocessor,
    rewrite_tablegen_definitions,
    rewrite_tablegen_classes,
    rewrite_tablegen_parent_class,
    rewrite_tablegen_strings,
    rewrite_tablegen_code_strings,
    rewrite_tablegen_values,
]
REWRITE_FUNCTIONS = (
    PYTHON_REWRITE_FUNCTIONS
    + CPP_REWRITE_FUNCTIONS
    + TABLEGEN_REWRITE_FUNCTIONS
    + CMAKE_REWRITE_FUNCTIONS
)


def is_in_range(node):
    if cli_args.line:
        line = cli_args.line
        range = cli_args.range
        range_start = line - range
        range_end = line + range

        node_start = node.start_point[0]
        if range_start <= node_start <= range_end:
            debug(
            f"{node.type} not in range in lines {cli_args.line - cli_args.range} - {cli_args.line + cli_args.range}"
                )
            return True
        else:
            return False
    else:
        return True


def traverse_node(node: tree_sitter.Node, indent=0):
    stack = [(node, indent)]
    while stack:
        node, indent = stack.pop()
        if is_in_range(node):
            print(f"{' ' * indent}{node.type} {node.start_point}")
            stack.extend(
                (field_child, indent + 2) for field_child in reversed(node.children)
            )
        else:
            stack.extend((field_child, 0) for field_child in reversed(node.children))


def parse_code(code, parser):
    tree = parser.parse(bytes(code, "utf8"))
    return tree


def traverse_code(code: str, parser: tree_sitter.Parser):
    tree = parse_code(code, parser)
    traverse_node(tree.root_node)


def find_code(find: str, code: str, parser: tree_sitter.Parser):
    tree = parse_code(code, parser)

    stack = [(tree.root_node, 0)]
    while stack:
        node, indent = stack.pop()

        if find and node is not tree.root_node:
            if find in get_node_str(node, code):
                print(f"Found {find}")
                print_node(node, code, show_tree=True)

        stack.extend(
            (field_child, indent + 2) for field_child in reversed(node.children)
        )


def get_node_str(node: tree_sitter.Node, code: str):
    start_byte = node.start_byte
    end_byte = node.end_byte
    return code[start_byte:end_byte]


def get_pretty_code(code: str):
    num_of_chars = 64
    code_lines = code.splitlines()
    if len(code_lines) > 1:
        first_line = code_lines[0][:num_of_chars]
        return first_line.strip() + "..."
    else:
        code_line = code.strip()
        if len(code_line) > num_of_chars:
            return code_line[:num_of_chars] + "..."
        else:
            return code_line


def get_node_str_pretty(node: tree_sitter.Node, code: str):
    node_str = get_node_str(node, code)
    return get_pretty_code(node_str)


def print_node(node: tree_sitter.Node, code: str, show_tree=False, pretty_print=True):
    if pretty_print:
        node_str = get_node_str_pretty(node, code)
    else:
        node_str = get_node_str(node, code)
    print(f"{node_str} ({node.type} at {node.start_point[0]})")
    if show_tree:
        traverse_node(node, indent=2)


def add_edit(
    node: tree_sitter.Node,
    code: str,
    edits: list,
    rewrite: tuple,
    strict=False,
    force=False,
):
    if not is_in_range(node):
        return

    if not can_rewrite(node, code, rewrite) and (not force):
        return
    else:
        debug(f"can rewrite {get_node_str_pretty(node, code)} with {rewrite}")

    node_str = get_node_str(node, code)
    src = rewrite[0]
    if strict:
        if src == node_str:
            edits.append(rewrite_node(node, code, rewrite))
        else:
            debug(f"no match between {src} and {node_str}")
    elif src in node_str:
        edits.append(rewrite_node(node, code, rewrite))


def apply_edits(code: str, edits: list):
    # sort edits in reverse order of position
    edits.sort(key=lambda edit: -edit[0])

    # apply edits
    for start, end, replacement in edits:
        code = code[:start] + replacement + code[end:]

    return code


def rewrite_node(node: tree_sitter.Node, code: str, rewrite: tuple):
    try:
        src, target, strict = rewrite
        node_code = code[node.start_byte : node.end_byte]
        if src in node_code:
            new_node_code = node_code.replace(src, target)
            info(
                f'rewrote "{get_pretty_code(node_code)}" -> "{get_pretty_code(new_node_code)}" [{node.type}] with {get_pretty_code(src), get_pretty_code(target)} at line {node.start_point[0]}. Called by {get_rewrite_function()}.'
            )
            return (node.start_byte, node.end_byte, new_node_code)
        else:
            raise Exception(
                f'Error: could not rewrite "{code[node.start_byte:node.end_byte]}" with {rewrite}'
            )
    except Exception as e:
        traceback.print_stack()
        raise e


def traverse_and_apply_fn(
    code: str, parser: tree_sitter.Parser, rewrite_fn, rewrite_map
):
    # parse cur code in to tree
    tree = parse_code(code, parser)

    edits = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()

        if is_in_range(node):
            rewrite_fn(node, code, edits, rewrite_map)

        stack.extend(reversed(node.children))

    # apply edits
    new_code = apply_edits(code, edits)
    return new_code


def filter_code(code, parser):
    print("filter_code")
    # Parse current code into a tree
    tree = parse_code(code, parser)

    # List to store the ranges (start and end bytes) of functions to exclude
    exclude_ranges = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()

        if is_in_range(node) and node.type == "function_definition":
            function_declarator = node.children[1] if len(node.children) > 1 else None
            if (
                function_declarator
                and function_declarator.type == "function_declarator"
            ):
                identifier = (
                    function_declarator.children[0]
                    if len(function_declarator.children) > 0
                    else None
                )
                if identifier and identifier.type == "identifier":
                    identifier_str = get_node_str(identifier, code)
                    if identifier_str not in FUNCTION_LIST:
                        exclude_ranges.append((node.start_byte, node.end_byte))

        stack.extend(reversed(node.children))

    # Generate the filtered code, excluding the ranges identified
    filtered_code = ""
    prev_end = 0
    for start, end in sorted(exclude_ranges):
        filtered_code += code[prev_end:start]
        prev_end = end
    filtered_code += code[prev_end:]

    return filtered_code


def read_file(file_path) -> str:
    # detect encoding
    with open(file_path, "rb") as file:
        raw_data = file.read()
        if raw_data:
            encoding = chardet.detect(raw_data)["encoding"]
        else:
            debug(f"Empty file {file_path} using utf-8")
            encoding = "utf-8"

    # read file with detected encoding
    with open(file_path, "r", encoding=encoding) as file:
        content = file.read()

    # convert to UTF-8 if necessary
    if encoding != "utf-8":
        debug(f"{file_path} contents converted to utf-8")
        content = content.encode(encoding).decode("utf-8")

    return content


def write_file(file_path, content):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return True
    except IOError as e:
        print(f"An error occurred: {e}")
        return False


def rewrite_file_impl(file_path, parser, rewrite_functions, rewrite_map):
    debug(f"rewritng {pretty_file_path(file_path)}")

    # read code
    code = read_file(file_path)

    # copy on the code we need
    # if FUNCTION_LIST:
    #     code = filter_code(code, parser)

    # traverse tree
    if cli_args.visualize or cli_args.line:
        traverse_code(code, parser)

    # find code
    if cli_args.find:
        find_code(cli_args.find, code, parser)
        return

    # rewrite code
    cur_code = code
    for rewrite_fn in rewrite_functions:
        cur_code = traverse_and_apply_fn(cur_code, parser, rewrite_fn, rewrite_map)

    # write back to file
    write_file(file_path, cur_code)

    # make sure file is editable
    if cli_args.chmod:
        chmod_recursive(
            file_path, cli_args.chmod
        )
    else:
        chmod_recursive(
            file_path, "660" # allow read and write
        )


def pretty_file_path(path: Path):
    try:
        new_path = path.relative_to(DEST_DIR)
    except ValueError:
        # This error is raised when the original path does not start with the remove_part
        raise ValueError(
            f"Cannot remove top part {DEST_DIR} as it's not a parent of {path}"
        )
    return new_path
    # return f"{path.parent.name}/{path.parent.name}/{path.name}"


def chmod_recursive(path, mode_str: str):
    mode = int(mode_str, 8)

    if os.path.isfile(path):
        os.chmod(path, mode)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            print(root, dirs, files)
            for dir in dirs:
                os.chmod(os.path.join(root, dir), mode)
            for file in files:
                os.chmod(os.path.join(root, file), mode)
    else:
        raise (Exception(f"Failed to change permissions for {path}"))


def rewrite_dir(dir_path: Path, rewrite_map: dict):
    # process each file in the destination directory
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = Path(os.path.join(root, file_name))
            rewrite_file(file_path, rewrite_map)


def combine_rewrite_rules(language_rewrite_map, file_rewrite_map):
    extended_rewrite_map = copy.deepcopy(language_rewrite_map)
    for rewrite_type, file_rewrite_rules in file_rewrite_map.items():
        if rewrite_type in extended_rewrite_map:
            # If the rewrite type is in language_rewrite_map, extend its list.
            extended_rewrite_map[rewrite_type].extend(file_rewrite_rules)
        else:
            # If the rewrite type is not in language_rewrite_map, add it.
            extended_rewrite_map[rewrite_type] = file_rewrite_rules

    return extended_rewrite_map


def rewrite_file(file_path: Path, file_rewrite_map: dict):
    if file_path.suffix in [".cpp", ".hpp", ".h", ".cc", ".c"]:
        rewrite_map = combine_rewrite_rules(TRANSLATION_MAP["cpp"], file_rewrite_map)
        rewrite_file_impl(file_path, CPP_PARSER, CPP_REWRITE_FUNCTIONS, rewrite_map)
    elif file_path.name == "CMakeLists.txt" or file_path.suffix in [".cmake"]:
        rewrite_map = combine_rewrite_rules(TRANSLATION_MAP["cmake"], file_rewrite_map)
        rewrite_file_impl(file_path, CMAKE_PARSER, CMAKE_REWRITE_FUNCTIONS, rewrite_map)
    elif file_path.suffix == ".td":
        rewrite_map = combine_rewrite_rules(
            TRANSLATION_MAP["tablegen"], file_rewrite_map
        )
        rewrite_file_impl(
            file_path, TABLEGEN_PARSER, TABLEGEN_REWRITE_FUNCTIONS, rewrite_map
        )
    elif file_path.suffix == ".py":
        rewrite_map = combine_rewrite_rules(TRANSLATION_MAP["python"], file_rewrite_map)
        rewrite_file_impl(
            file_path, PYTHON_PARSER, PYTHON_REWRITE_FUNCTIONS, rewrite_map
        )
    else:
        error(f"unable to process {file_path}")


def rewrite_files_impl(rewrite_maps: List[dict], progress_bar = True):
    for rewrite_map in tqdm(rewrite_maps, disable=not progress_bar):
        src_path = rewrite_map["src"]
        dest_path = rewrite_map["dest"]
        if "line" in rewrite_map:
            cli_args.line = rewrite_map["line"]
        if "range" in rewrite_map:
            cli_args.range = rewrite_map["range"]
        if "map" not in rewrite_map:
            rewrite_map["map"] = {}

        if Path.is_file(src_path) and Path.exists(src_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            rewrite_file(dest_path, rewrite_map["map"])
        else:
            raise TypeError(f"Error: bad rewrite request {rewrite_map}")


def rewrite_files_parallel(rewrite_maps: List[dict]):
    # split map
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    chunks = [rewrite_maps[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(lambda chunk: rewrite_files_impl(chunk, progress_bar=False), chunks)


def rewrite_files(rewrite_maps: List[dict],):
    if cli_args.parallel:
        rewrite_files_parallel(rewrite_maps)
    else:
        rewrite_files_impl(rewrite_maps)



def is_path_inside(child_path: Path, parent_path: Path) -> bool:
    try:
        child_abs_path = child_path.resolve()
        parent_abs_path = parent_path.resolve()
        child_abs_path.relative_to(parent_abs_path)
        return True
    except ValueError:
        return False

def clone_repos(repo_list, clone_dir = "/tmp"):
    clone_dir = Path(clone_dir)
    for repo_url in repo_list:
        repo_name = repo_url.split("/")[-1]
        repo_path = clone_dir.joinpath(repo_name)
        if not os.path.isdir(repo_path):
            try:
                Repo.clone_from(repo_url, repo_path)
                info(f"Cloned {repo_name} to {repo_path}")
            except GitCommandError as e:
                error(f"Failed to clone {repo_url}: {e}")
        else:
            debug(f"{repo_name} already exists at {repo_path}")

def build_tree_sitter(build_path="build"):
    tmp_dir = Path("/tmp")

    # clone language parsers
    clone_repos(
        [
            "https://github.com/tree-sitter/tree-sitter-python",
            "https://github.com/tree-sitter/tree-sitter-cpp",
            "https://github.com/uyha/tree-sitter-cmake",
            "https://github.com/Flakebi/tree-sitter-tablegen",
        ],
        clone_dir=tmp_dir.as_posix()
    )

    # load languages
    shared_lib_path = Path(build_path).joinpath("tree-sitter-languages.so")
    tree_sitter.Language.build_library(
       shared_lib_path.as_posix(),
        [
            tmp_dir.joinpath("tree-sitter-python"),
            tmp_dir.joinpath("tree-sitter-cpp"),
            tmp_dir.joinpath("tree-sitter-cmake"),
            tmp_dir.joinpath("tree-sitter-tablegen"),
        ],
    )

    return shared_lib_path


def rewrite_src_path(src_path: str, paths_to_rename: dict) -> Path:
    dirs_to_rename = paths_to_rename["dirs"]
    files_to_rename = paths_to_rename["files"]

    path = Path(src_path)
    parts = list(path.parts)

    # ignore filenames for now
    has_filename = path.suffix != ""
    if has_filename:
        dir_parts = parts[:-1]
        file_name = parts[-1]
    else:
        dir_parts = parts
        file_name = None

    # rewrite path
    dir_path_str = "/".join(dir_parts)
    for original, new, strict in dirs_to_rename:
        if strict:
            if original == dir_path_str:
                debug(f"rewrite dir {dir_path_str} with {(original, new, strict)}")
                dir_path_str = dir_path_str.replace(original, new)
        else:
            if original in dir_path_str:
                debug(f"rewrite dir {dir_path_str} with {(original, new, strict)}")
                dir_path_str = dir_path_str.replace(original, new)

    # reconstruct the path
    new_dir_parts = dir_path_str.split("/")
    if file_name:
        for original, new, strict in files_to_rename:
            if strict:
                if original == file_name:
                    debug(f"rewrite file_name {file_name} with {(original, new, strict)}")
                    file_name = file_name.replace(original, new)
            else:
                if original in file_name:
                    debug(f"rewrite file_name {file_name} with {(original, new, strict)}")
                    file_name = file_name.replace(original, new)

        new_dir_parts.append(file_name)
    new_path = Path(*new_dir_parts)
    return new_path





def get_all_files_in_directory(directory_path):
    """Return a list of file paths for all files in a given directory and its subdirectories."""
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def should_ignore_path(path: Path):
    ignored_paths = ["__pycache__", "third_party", "_C"]
    for ip in ignored_paths:
        if ip in str(path):
            debug(f"ignored {path} because it contains {ip}")
            return True

    return False

def find_src_path(dest_path: str) -> str:
    # find src path
    relative_path = str(Path(dest_path).absolute().relative_to(DEST_DIR))
    inverse_dict = {}
    inverse_dict["dirs"] = [
        (v, k, strict) for k, v, strict in TRANSLATION_MAP["paths"]["dirs"]
    ]
    inverse_dict["files"] = [
        (v, k, strict) for k, v, strict in TRANSLATION_MAP["paths"]["files"]
    ]
    src_relative_path = rewrite_src_path(relative_path, inverse_dict)
    ret = str(src_relative_path)
    if src_relative_path.exists():
        info(f"found original source path {ret}")
        return ret
    else:
        raise ValueError(f"Error: Could not find original source path {ret}")

def process_file_rewrite_request(rewrite_request: dict, rewrite_maps: list):
    debug(f"rewrite_request: {rewrite_request}")

    # process src path
    user_src_path = rewrite_request["src"]
    user_src_abs_path = SRC_DIR.joinpath(user_src_path).absolute()
    if not Path.exists(user_src_abs_path):
        raise ValueError(f"{user_src_abs_path} doesnot exist!")
    elif is_path_inside(user_src_abs_path, DEST_DIR):
        warning(f"{user_src_path} in the destination dir {DEST_DIR}")
        src_path = find_src_path(user_src_path)
        src_abs_path = SRC_DIR.joinpath(src_path).absolute()
    else:
        src_path = user_src_path
        src_abs_path = user_src_abs_path
    
    # prcess dest path
    if rewrite_request["dest"]:
        dest_path = rewrite_request["dest"]
    else:
        dest_path = rewrite_src_path(
            src_path, TRANSLATION_MAP["paths"]
        )
    dest_abs_path = DEST_DIR.joinpath(dest_path).absolute()

    # append to list of rewrites
    rewrite_request["src"] = src_abs_path
    rewrite_request["dest"] = dest_abs_path
    rewrite_maps.append(rewrite_request)
    debug(f"{src_abs_path.relative_to(SRC_DIR)} -> {dest_abs_path.relative_to(DEST_DIR)}")


def process_rewrite_request(rewrite_request: dict, rewrite_maps: list):
    src_path = Path(rewrite_request["src"])
    if Path.is_file(src_path):
        process_file_rewrite_request(rewrite_request, rewrite_maps)
    elif Path.is_dir(src_path):
        dir_files = get_all_files_in_directory(rewrite_request["src"])
        for file_path in dir_files:
            # create dest file path
            if rewrite_request["dest"]:
                dest_file_path = file_path.replace(
                    rewrite_request["src"], rewrite_request["dest"]
                )
            else:
                dest_file_path = None  # this will generate a file path

            # rewrite file
            if not should_ignore_path(file_path):
                file_rewrite_request = {
                    "src": file_path,
                    "dest": dest_file_path,
                    "map": {},
                }
                process_file_rewrite_request(file_rewrite_request, rewrite_maps)
    else:
        raise ValueError(f"invalid rewrite request {rewrite_request}")


def get_rewrite_maps(rewrite_list: list):
    rewrite_maps: list[Dict] = []
    for rewrite_request in tqdm(rewrite_list, disable=True):
        if type(rewrite_request) == str:
            rewrite_request = {
                "src": rewrite_request,
                "dest": None,
                "map": {},
            }
            process_rewrite_request(rewrite_request, rewrite_maps)
        elif type(rewrite_request) == tuple:
            rewrite_request = {
                "src": rewrite_request[0],
                "dest": rewrite_request[1],
                "map": {},
            }
            process_rewrite_request(rewrite_request, rewrite_maps)
        elif type(rewrite_request) == dict:
            if "src" in rewrite_request.keys() and "dest" in rewrite_request.keys():
                process_rewrite_request(rewrite_request, rewrite_maps)
            else:
                raise ValueError(
                    f"invalid rewrite_request {rewrite_request}. Must contain a src and a dest."
                )
        else:
            raise TypeError(f"{rewrite_request} is not a str, tuple or dict")

    # user defined filter
    if cli_args.filter:
        user_filtered_rewrite_maps = []
        for rm in rewrite_maps:
            if cli_args.filter in str(rm["src"]):
                user_filtered_rewrite_maps.append(rm)
        info(
            f"filtered rewrite map list from {len(rewrite_maps)} to {len(user_filtered_rewrite_maps)} with {cli_args.filter}"
        )
        rewrite_maps = user_filtered_rewrite_maps

    return rewrite_maps


def parse_args():
    def parse_comma_separated(input_string):
        return tuple(input_string.split(","))

    def json_file_or_string(json_input):
        if os.path.isfile(json_input):
            with open(json_input, "r") as file:
                return json.load(file)
        try:
            return json.loads(json_input)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError(f"Invalid JSON input: {json_input}")

    # args
    parser = argparse.ArgumentParser(
        description="Run the script with optional debug mode."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--quiet", action="store_true", help="minimal ouput")
    parser.add_argument("--verbose", action="store_true", help="Enable more output")
    parser.add_argument("--time", action="store_true", help="time run")
    parser.add_argument("--parallel", action="store_true", help="run in parallel")
    parser.add_argument(
        "--visualize", action="store_true", help="visualize the tree for the program"
    )
    parser.add_argument("--path", type=str, help="Path to source code")
    parser.add_argument(
        "--filter", type=str, help="process files only with this string"
    )
    parser.add_argument(
        "--chmod", type=str, help="permission to apply to files that are gened"
    )
    parser.add_argument(
        "--find", type=str, help="Find the node associated with some code"
    )
    parser.add_argument(
        "--line", type=int, help="see the rewrites applied to a certain line"
    )
    parser.add_argument(
        "--range",
        type=int,
        default=1,
        help="see the rewrites applied to a certain line",
    )
    parser.add_argument(
        "--paths", type=parse_comma_separated, help="src_path,dest_path"
    )
    parser.add_argument(
        "--map", type=json_file_or_string, help="JSON file path or JSON string"
    )
    return parser.parse_args()


if __name__ == "__main__":
    CURRENT_PATH = Path(__file__).absolute()
    CURRENT_DIR = CURRENT_PATH.parents[0]
    ROOT_DIR = CURRENT_PATH.parents[2]
    SRC_DIR = ROOT_DIR
    DEST_DIR = ROOT_DIR.joinpath("python", "triton", "third_party", "hip", "gen")

    # cli
    cli_args = parse_args()

    # logger
    root = logging.getLogger()
    if cli_args.quiet:
        root.setLevel(logging.ERROR)
    elif cli_args.debug:
        root.setLevel(logging.DEBUG)
    elif cli_args.verbose:
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.WARN)

    language_lib_path = build_tree_sitter().as_posix()

    PYTHON_LANGUAGE = tree_sitter.Language(language_lib_path, "python")
    PYTHON_PARSER = tree_sitter.Parser()
    PYTHON_PARSER.set_language(PYTHON_LANGUAGE)

    CPP_LANGUAGE = tree_sitter.Language(language_lib_path, "cpp")
    CPP_PARSER = tree_sitter.Parser()
    CPP_PARSER.set_language(CPP_LANGUAGE)

    CMAKE_LANGUAGE = tree_sitter.Language(language_lib_path, "cmake")
    CMAKE_PARSER = tree_sitter.Parser()
    CMAKE_PARSER.set_language(CMAKE_LANGUAGE)

    TABLEGEN_LANGUAGE = tree_sitter.Language(language_lib_path, "tablegen")
    TABLEGEN_PARSER = tree_sitter.Parser()
    TABLEGEN_PARSER.set_language(TABLEGEN_LANGUAGE)

    # load translation map
    TRANSLATION_MAP = {  # TODO: Improve or Unifiy this appraoch not the best. Rewrites is just not data. Some nodes rewrite requires code
        "paths": {
            "dirs": [
                (
                    "include/triton/Dialect/Triton",
                    "include/triton/Dialect/TritonROCM",
                    True,
                ),  # handles the case where we specifies just the folder
                ("lib/Dialect/Triton", "lib/Dialect/TritonROCM", True),
                (
                    "include/triton/Dialect/Triton/",
                    "include/triton/Dialect/TritonROCM/",
                    False,
                ),
                ("lib/Dialect/Triton/", "lib/Dialect/TritonROCM/", False),
                ("TritonGPU", "TritonGPUROCM", False),
                ("Analysis", "AnalysisROCM", False),
                ("Sys", "SysROCM", False),
                ("LLVMIR", "LLVMROCMIR", False),
                ("python/src", ".", False),
                ("TritonNvidiaGPU", "TritonNvidiaGPUROCM", False),
                ("NVGPU", "NVGPUROCM", False),
                ("PTX", "PTXROCM", False),
                ("python/triton", "triton_rocm", False),
            ],
            "files":
            [
                ("MLIRTypes.h", "MLIRTypesROCM.h", False)
            ]
        },
        "python": {
            "imports": [
                (
                    "libtriton",
                    "librocm_backend_for_triton",
                    True,
                ),
                ("triton", "triton.third_party.hip.gen.python.triton_rocm", True),
            ],
            "relative_imports": [
                (
                    "libtriton",
                    "librocm_backend_for_triton",
                    True,
                ),
                ("triton", "triton_rocm", True),
            ],
            "types": [
                ("JITFunction", "JITFunctionROCM", True),
                ("triton", "triton.third_party.hip.gen.python.triton_rocm", True),
            ],
            "functions":[
                ("parse_mlir_module", "parse_mlir_module_rocm", False),
                ("ttir_to_ttgir", "ttir_to_ttgir_rocm", True),
                ("ttgir_to_llir", "ttgir_to_llir_rocm", True),
                ("optimize_ttgir", "optimize_ttgir_rocm", True),
                ("optimize_ttir", "optimize_ttir_rocm", True),
                ("ast_to_ttir", "ast_to_ttir_rocm", True),
            ],
            "strings":
            [
                ("JITFunction", "JITFunctionROCM", False),
            ]
        },
        "cpp": {
            "preprocessor": [["USE_ROCM", 1]],
            "defines": [
                ("TRITON_ANALYSIS", "TRITON_ANALYSISROCM", False),
                ("TRITON_CONVERSION", "TRITON_CONVERSION_ROCM", False),
                ("TRITON_DIALECT_TRITONGPU", "TRITON_DIALECT_TRITONGPUROCM", False),
                ("TRITON_DIALECT_TRITON_", "TRITON_DIALECT_TRITONROCM_", False),
                (
                    "TRITON_DIALECT_TRITONNVIDIAGPU",
                    "TRITON_DIALECT_TRITONNVIDIAGPUROCM",
                    False,
                ),
                ("TRITON_IR", "TRITONROCM_IR", False),
                ("TRITON_GPU_IR_", "TRITON_GPU_ROCM_IR_", False),
                ("TRITONGPU_IR_", "TRITONGPUROCM_IR_", False),
                ("TRITONGPU_CONVERSION", "TRITONGPUROCM_CONVERSION", False),
                ("TRITON_TARGET_LLVM_IR_", "TRITON_TARGET_LLVM_IR_ROCM_", False),
                ("TDL_TOOLS_SYS_", "TDL_TOOLS_SYS_ROCM_", False),
                (
                    "PASS_DEF_CONVERTTRITONGPUTOLLVM",
                    "PASS_DEF_CONVERTTRITONGPUROCMTOLLVM",
                    False,
                ),
                ("TRITON_TARGET_PTX", "TRITON_TARGET_PTXROCM", False),
            ],
            "preproc_args": [
                ("triton::", "triton_rocm::", False),
            ],
            "includes": [
                ("/Triton/", "/TritonROCM/", False),
                ("/TritonGPU/", "/TritonGPUROCM/", False),
                ("/TritonNvidiaGPU/", "/TritonNvidiaGPUROCM/", False),
                ("/NVGPU/", "/NVGPUROCM/", False),
                ("triton/Analysis/", "triton/AnalysisROCM/", False),
                ("/TritonToTritonGPU/", "/TritonToTritonGPUROCM/", False),
                ("/TritonGPUToLLVM/", "/TritonGPUROCMToLLVM/", False),
                ("/NVGPUToLLVM/", "/NVGPUROCMToLLVM/", False),
                ("/PTX/", "/PTXROCM/", False),
                ("/Tools/Sys/", "/Tools/SysROCM/", False),
                ("triton/Target/LLVMIR", "triton/Target/LLVMROCMIR", False),
                ("Conversion/MLIRTypes.h", "Conversion/MLIRTypesROCM.h", False),
            ],
            "namespaces": [
                # ("triton::gpu::TritonGPU", "triton::gpu_rocm::TritonGPUROCM", False),
                ("TritonGPU", "TritonGPUROCM", False),
                ("TritonDialect", "TritonROCMDialect", False),
                ("TritonNvidiaGPUDialect", "TritonNvidiaGPUROCMDialect", False),
                ("NVGPU", "NVGPUROCM", False),
                ("gpu", "gpu_rocm", True),
                ("triton", "triton_rocm", True),
            ],
            "types": [
                ("TritonDialect", "TritonROCMDialect", True),
                ("TritonGPU", "TritonGPUROCM", False),
                ("TritonNvidiaGPUDialect", "TritonNvidiaGPUROCMDialect", False),
                ("NVGPU", "NVGPUROCM", False),
                ("ResultsAreSharedEncoding", "ResultsAreSharedEncodingROCM", False),
                ("maxTensorNumElements", "maxTensorNumElementsROCM", False),
                ("SameOperandsEncoding", "SameOperandsEncodingROCM", False),
                (
                    "SameOperandsAndResultEncoding",
                    "SameOperandsAndResultEncodingROCM",
                    False,
                ),
                ("SameLoadStoreOperandsShape", "SameLoadStoreOperandsShapeROCM", False),
                (
                    "SameLoadStoreOperandsAndResultEncoding",
                    "SameLoadStoreOperandsAndResultEncodingROCM",
                    False,
                ),
                ("TensorSizeTrait", "TensorSizeTraitROCM", False),
                (
                    "SameLoadStoreOperandsEncoding",
                    "SameLoadStoreOperandsEncodingROCM",
                    False,
                ),
                (
                    "SameLoadStoreOperandsAndResultShape",
                    "SameLoadStoreOperandsAndResultShapeROCM",
                    False,
                ),
            ],
            "functions": [
                ("TritonGPU", "TritonGPUROCM", False),
                ("TritonNvidiaGPU", "TritonNvidiaGPUROCM", False),
                ("TritonDialect", "TritonROCMDialect", False),
                ("NVGPU", "NVGPUROCM", False),
                (
                    "verifyResultsAreSharedEncoding",
                    "verifyResultsAreSharedEncodingROCM",
                    False,
                ),
                ("SameOperandsEncoding", "SameOperandsEncodingROCM", False),
                (
                    "SameOperandsAndResultEncoding",
                    "SameOperandsAndResultEncodingROCM",
                    False,
                ),
                ("SameLoadStoreOperandsShape", "SameLoadStoreOperandsShapeROCM", False),
                (
                    "SameLoadStoreOperandsAndResultEncoding",
                    "SameLoadStoreOperandsAndResultEncodingROCM",
                    False,
                ),
                ("TensorSizeTrait", "TensorSizeTraitROCM", False),
                (
                    "SameLoadStoreOperandsEncoding",
                    "SameLoadStoreOperandsEncodingROCM",
                    False,
                ),
                (
                    "SameLoadStoreOperandsAndResultShape",
                    "SameLoadStoreOperandsAndResultShapeROCM",
                    False,
                ),
            ],
            "strings": [
                # ("triton", "triton_rocm", True),
                # ("triton_gpu.", "triton_gpu_rocm.", False)
            ],
        },
        "tablegen": {
            "defs": [
                ("ResultsAreSharedEncoding", "ResultsAreSharedEncodingROCM", False),
                ("Triton_Dialect", "TritonROCM_Dialect", False),
                ("TT_", "TTROCM_", False),
                ("TTNG_", "TTNGROCM_", False),
                ("NVGPU_", "NVGPUROCM_", False),
                ("TritonNvidiaGPU_Dialect", "TritonNvidiaGPUROCM_Dialect", False),
                ("ConvertNVGPUToLLVM", "ConvertNVGPUROCMToLLVM", False),
            ],
            "parent_class_list": [
                ("TritonGPU", "TritonGPUROCM", False),
                ("ResultsAreSharedEncoding", "ResultsAreSharedEncodingROCM", False),
                ("TritonTypeDef", "TritonROCMTypeDef", False),
                ("TT_", "TTROCM_", False),
                ("TTNG_", "TTNGROCM_", False),
                ("NVGPU_", "NVGPUROCM_", False),
                ("Triton_Dialect", "TritonROCM_Dialect", False),
                ("TritonNvidiaGPU_Dialect", "TritonNvidiaGPUROCM_Dialect", False),
            ],
            "classes": [
                ("TT_", "TTROCM_", False),
                ("TTNG_", "TTNGROCM_", False),
                ("NVGPU_", "NVGPUROCM_", False),
            ],
            "values": [
                ("TT_", "TTROCM_", False),
                ("TTNG_", "TTNGROCM_", False),
                ("NVGPU_", "NVGPUROCM_", False),
                ("SameOperandsEncoding", "SameOperandsEncodingROCM", False),
                (
                    "SameOperandsAndResultEncoding",
                    "SameOperandsAndResultEncodingROCM",
                    False,
                ),
                ("SameLoadStoreOperandsShape", "SameLoadStoreOperandsShapeROCM", False),
                (
                    "SameLoadStoreOperandsAndResultEncoding",
                    "SameLoadStoreOperandsAndResultEncodingROCM",
                    False,
                ),
                ("TensorSizeTrait", "TensorSizeTraitROCM", False),
                (
                    "SameLoadStoreOperandsEncoding",
                    "SameLoadStoreOperandsEncodingROCM",
                    False,
                ),
                (
                    "SameLoadStoreOperandsAndResultShape",
                    "SameLoadStoreOperandsAndResultShapeROCM",
                    False,
                ),
            ],
            "strings": [
                ("triton::CacheModifier", "triton_rocm::CacheModifier", False),
                ("triton::EvictionPolicy", "triton_rocm::EvictionPolicy", False),
                ("triton::PaddingOption", "triton_rocm::PaddingOption", False),
                (
                    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                    "mlir::triton_rocm::nvidia_gpu::TritonNvidiaGPUROCMDialect",
                    False,
                ),
                ("mlir::triton::PointerType", "mlir::triton_rocm::PointerType", False),
                ("mlir::triton::Target", "mlir::triton_rocm::Target", False),
                ("mlir::createTritonGPU", "mlir::createTritonGPUROCM", False),
                (
                    "mlir::createTritonNvidiaGPU",
                    "mlir::createTritonNvidiaGPUROCM",
                    False,
                ),
                (
                    "mlir::triton::createTritonNvidiaGPU",
                    "mlir::triton::createTritonNvidiaGPUROCM",
                    False,
                ),
                (
                    "mlir::triton::createConvertNVGPU",
                    "mlir::triton_rocm::createConvertNVGPUROCM",
                    False,
                ),
                (
                    "mlir::triton::createConvertTritonGPU",
                    "mlir::triton_rocm::createConvertTritonGPUROCM",
                    False,
                ),
                (
                    "mlir::triton::createConvertTritonToTritonGPU",
                    "mlir::triton_rocm::createConvertTritonToTritonGPUROCM",
                    False,
                ),
                (
                    "mlir::triton::nvgpu::NVGPUDialect",
                    "mlir::triton_rocm::nvgpu::NVGPUROCMDialect",
                    False,
                ),
                (
                    "mlir::triton::createCombineOpsPass",
                    "mlir::triton_rocm::createCombineOpsPass",
                    False,
                ),
                (
                    "mlir::triton::createReorderBroadcastPass",
                    "mlir::triton_rocm::createReorderBroadcastPass",
                    False,
                ),
                (
                    "mlir::triton::createRewriteTensorPointerPass",
                    "mlir::triton_rocm::createRewriteTensorPointerPass",
                    False,
                ),
                ('"::mlir::triton"', '"::mlir::triton_rocm"', True),
                ('"::mlir::triton::gpu"', '"::mlir::triton_rocm::gpu_rocm"', True),
                (
                    '"::mlir::triton::nvidia_gpu"',
                    '"::mlir::triton_rocm::nvidia_gpu"',
                    True,
                ),
                ('"::mlir::triton::nvgpu"', '"::mlir::triton_rocm::nvgpu"', False),
                ('"triton::TritonDialect"', '"triton_rocm::TritonROCMDialect"', True),
                (
                    '"mlir::triton::TritonDialect"',
                    '"mlir::triton_rocm::TritonROCMDialect"',
                    True,
                ),
                (
                    "triton::gpu::TritonGPU",
                    "triton_rocm::gpu_rocm::TritonGPUROCM",
                    False,
                ),
                # ("triton_gpu", "triton_gpu_rocm", False),
                ("triton-gpu-", "triton-gpurocm-", False),
            ],
            "code_strings": [
                # ("triton_gpu", "triton_gpu_rocm", False),
                ("triton::gpu", "triton_rocm::gpu_rocm", False),
                ("Triton GPU", "Triton GPU ROCM", False),
                ("#ifdef USE_ROCM", "#if 1", False),
                ("#ifndef USE_ROCM", "#if 0", False),
                ("TritonDialect", "TritonROCMDialect", False),
            ],
            "preprocessor": [
                ("TRITONGPU", "TRITONGPUROCM", False),
                ("TRITON_CONVERSION", "TRITON_CONVERSION_ROCM", False),
                ("TRITON_ATTR", "TRITONROCM_ATTR", False),
                ("TRITON_DIALECT", "TRITONROCM_DIALECT", False),
                ("TRITON_INTERFACES", "TRITONROCM_INTERFACES", False),
                ("TRITON_OPS", "TRITONROCM_OPS", False),
                ("TRITON_TYPES", "TRITONROCM_TYPES", False),
                ("TRITON_PASSES", "TRITONROCM_PASSES", False),
                ("TRITONNVIDIAGPU_DIALECT", "TRITONNVIDIAGPUROCM_DIALECT", False),
                ("TRITONNVIDIAGPU_TYPES", "TRITONNVIDIAGPUROCM_TYPES", False),
                ("TRITONNVIDIAGPU_PASSES", "TRITONNVIDIAGPUROCM_PASSES", False),
                ("TRITONNVIDIAGPU_OPS", "TRITONNVIDIAGPUROCM_OPS", False),
                ("NVGPU_", "NVGPUROCM_", False),
            ],
            "include": [
                ("/TritonGPU/", "/TritonGPUROCM/", False),
                ("/Triton/", "/TritonROCM/", False),
                ("/TritonNvidiaGPU/", "/TritonNvidiaGPUROCM/", False),
                ("/NVGPU/", "/NVGPUROCM/", False),
            ],
        },
        "cmake": {
            "argument": [
                ("TritonNvidiaGPUIR", "TritonNvidiaGPUROCMIR", True),
                ("TritonNvidiaGPUTableGen", "TritonNvidiaGPUROCMTableGen", True),
                (
                    "TritonNvidiaGPUAttrDefsIncGen",
                    "TritonNvidiaGPUROCMAttrDefsIncGen",
                    True,
                ),
                ("TritonNvidiaGPUOps", "TritonNvidiaGPUROCMOps", True),
                ("TritonNvidiaGPUDialect", "TritonNvidiaROCMGPUDialect", False),
                ("TritonNvidiaGPUTransforms", "TritonNvidiaGPUROCMTransforms", True),
                (
                    "TritonNvidiaGPUTransformsIncGen",
                    "TritonNvidiaGPUROCMTransformsIncGen",
                    True,
                ),
                ("TritonNvidiaGPU", "TritonNvidiaGPUROCM", True),
                ("Triton", "TritonROCM", True),
                ("TritonDialect", "TritonROCMDialect", True),
                ("TritonOps", "TritonROCMOps", True),
                ("TritonTableGen", "TritonROCMTableGen", True),
                ("TritonTransformsIncGen", "TritonROCMTransformsIncGen", True),
                # ("triton_gpu", "triton_gpu_rocm", False),
                ("TritonGPU", "TritonGPUROCM", True),
                ("TritonTransforms", "TritonROCMTransforms", True),
                ("TritonCombineIncGen", "TritonROCMCombineIncGen", True),
                ("TritonIR", "TritonROCMIR", True),
                ("TritonGPUIR", "TritonGPUROCMIR", True),
                ("TritonGPUToLLVM", "TritonGPUROCMToLLVM", True),
                ("TritonToTritonGPU", "TritonToTritonGPUROCM", True),
                ("TritonAnalysis", "TritonAnalysisROCM", True),
                ("TritonGPUTableGen", "TritonGPUROCMTableGen", True),
                ("TritonGPUAttrDefsIncGen", "TritonGPUROCMAttrDefsIncGen", True),
                ("TritonGPUTransformsIncGen", "TritonGPUROCMTransformsIncGen", True),
                ("TritonGPUOps", "TritonGPUROCMOps", True),
                ("TritonGPUAttrDefs", "TritonGPUROCMAttrDefs", True),
                ("TritonGPUDialect", "TritonGPUROCMDialect", True),
                ("TritonGPUTransforms", "TritonGPUROCMTransforms", True),
                (
                    "TritonConversionPassIncGen",
                    "TritonToTritonGPUROCMConversionPassIncGen",
                    True,
                ),
                (
                    "TritonGPUConversionPassIncGen",
                    "TritonGPUROCMConversionPassIncGen",
                    True,
                ),
                ("ASMBuilder", "GCNASMBuilder", True),
                ("/TritonGPUToLLVM", "/TritonGPUROCMToLLVM", False),
                ("/TritonToTritonGPU", "/TritonToTritonGPUROCM", False),
                ("/NVGPUToLLVM", "/NVGPUROCMToLLVM", False),
                ("LLVMIR", "LLVMROCMIR", True),
                ("TritonLLVMIR", "TritonLLVMROCMIR", True),
                ("LLVMIRIncGen", "LLVMROCMIRIncGen", True),
                ("NVGPUConversionPassIncGen", "NVGPUROCMConversionPassIncGen", True),
                ("NVGPUToLLVM", "NVGPUROCMToLLVM", True),
                ("NVGPUTableGen", "NVGPUROCMTableGen", True),
                ("NVGPUAttrDefsIncGen", "NVGPUROCMAttrDefsIncGen", True),
                ("NVGPUOps", "NVGPUROCMOps", True),
                ("NVGPUDialect", "NVGPUROCMDialect", True),
                ("NVGPUIR", "NVGPUROCMIR", True),
                ("TritonPTX", "TritonPTXROCM", True),
                ("NVGPU", "NVGPUROCM", True),
                ("PTX", "PTXROCM", True),
                ("Analysis", "AnalysisROCM", True)
                # "MLIRGPUOps": "MLIRGPUDialect", # NOTE: disabled for now. enable when fork catches up to upstream
            ]
        },
    }

    # run rewrite
    user_rewrite_list = []
    FUNCTION_LIST = {}
    if cli_args.path:
        user_rewrite_list.append(cli_args.path)
    elif cli_args.paths:
        src_path = cli_args.paths[0]
        dest_path = cli_args.paths[1]
        user_rewrite_list.append((src_path, dest_path))
        FUNCTION_LIST = {
            "init_triton": "init_triton",
            "init_triton_translation": "init_triton_translation",
        }
    elif cli_args.map:
        user_rewrite_list.extend(cli_args.map)
    else:
        user_rewrite_list.extend(
            [
                "include",
                "lib",
                {"src": "python/triton", "dest": "python/triton_rocm"},
                {
                    "src": "python/src/main.cc",
                    "dest": "./rocm_backend_for_triton.cc",
                    "map": {
                        "functions": [
                            ["init_triton", "init_rocm_backend_for_triton", True]
                        ],
                        "types": [["libtriton", "librocm_backend_for_triton", True]],
                    },
                },
                {
                    "src": "python/src/triton.cc",
                    "dest": "./triton_rocm.cc",
                    "map": {
                        "functions": [
                            ["init_triton", "init_rocm_backend_for_triton", True]
                        ],
                        "types": [],
                        "includes": [],
                        "strings": [
                            ["triton", "triton_rocm", True],
                            ["parse_mlir_module", "parse_mlir_module_rocm", True],
                        ],
                        "dialect_removal": True,
                    },
                },
                
            ]
        )

    info("processing rewrite requests ...")
    debug(f"user_rewrite_list: {user_rewrite_list}")
    rewrite_maps = get_rewrite_maps(user_rewrite_list)
    debug(f"rewrote_maps: {rewrite_maps}")

    info("rewriting files")
    if cli_args.time:
        start_time = time.time()
        rewrite_files(rewrite_maps) 
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
    else:
        rewrite_files(rewrite_maps) 
