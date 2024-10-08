import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

vgpr_single_pattern = re.compile(r'(?<!\[)v(\d+)(?![\d:])')
vgpr_range_pattern = re.compile(r'v\[(\d+):(\d+)\]')
sgpr_single_pattern = re.compile(r'(?<!\[)s(\d+)(?![\d:])')
sgpr_range_pattern = re.compile(r's\[(\d+):(\d+)\]')

spill_pattern = re.compile(r'^scratch_store_dwordx?\d*')
reload_pattern = re.compile(r'^scratch_load_dwordx?\d*')
global_load_pattern = re.compile(r'^global_load_.*')
global_store_pattern = re.compile(r'^global_store_.*')
v_mfma_pattern = re.compile(r'^v_mfma_.*')

def parse_register_usage(lines):
    block_registers = defaultdict(lambda: {
        'vgprs': set(),
        'sgprs': set(),
        'vgpr_instruction_types': defaultdict(set),  # key: vgpr number, value: set of instruction types
    })
    current_block = None

    for line in lines:
        line = line.strip()

        label_match = re.match(r'^(\.\w+|; %bb\.\d+):', line)
        if label_match:
            current_block = label_match.group(1)
            continue

        if current_block:
            code = line.split(';')[0].strip()
            if not code:
                continue

            tokens = code.split()
            if not tokens:
                continue

            instruction = tokens[0]

            if spill_pattern.match(instruction):
                instruction_type = 'scratch_store'
            elif reload_pattern.match(instruction):
                instruction_type = 'scratch_load'
            elif v_mfma_pattern.match(instruction):
                instruction_type = 'v_mfma'
            elif global_load_pattern.match(instruction):
                instruction_type = 'global_load'
            elif global_store_pattern.match(instruction):
                instruction_type = 'global_store'
            else:
                instruction_type = 'others'

            vgprs_used = extract_vgprs_from_instruction(line)
            for vgpr in vgprs_used:
                block_registers[current_block]['vgprs'].add(vgpr)
                block_registers[current_block]['vgpr_instruction_types'][vgpr].add(instruction_type)

            sgpr_ranges = sgpr_range_pattern.findall(line)
            for sgpr_range in sgpr_ranges:
                start, end = int(sgpr_range[0]), int(sgpr_range[1])
                block_registers[current_block]['sgprs'].update(range(start, end + 1))

            line_no_sgpr_ranges = sgpr_range_pattern.sub('', line)

            sgprs = sgpr_single_pattern.findall(line_no_sgpr_ranges)
            for sgpr in sgprs:
                block_registers[current_block]['sgprs'].add(int(sgpr))

    return block_registers

def extract_vgprs_from_instruction(line):
    code = line.split(';')[0]
    tokens = code.strip().split()
    if len(tokens) < 2:
        return set()

    operands_str = ' '.join(tokens[1:])

    vgpr_ranges = vgpr_range_pattern.findall(operands_str)
    vgprs = set()
    for vgpr_range in vgpr_ranges:
        start, end = int(vgpr_range[0]), int(vgpr_range[1])
        vgprs.update(range(start, end + 1))

    operands_no_vgpr_ranges = vgpr_range_pattern.sub('', operands_str)

    vgprs.update(map(int, vgpr_single_pattern.findall(operands_no_vgpr_ranges)))
    return vgprs

def generate_call_tree_and_register_usage(assembly_file):

    with open(assembly_file, 'r') as file:
        all_lines = file.readlines()

    last_endpgm_index = None
    for idx, line in enumerate(all_lines):
        if 's_endpgm' in line:
            last_endpgm_index = idx

    if last_endpgm_index is None:
        print("No s_endpgm found in the assembly file.")
        return

    lines_to_process = all_lines[:last_endpgm_index + 1]

    call_tree = nx.DiGraph()

    label_pattern = re.compile(r'^(\.\w+|; %bb\.\d+):')
    jump_pattern = re.compile(r'(s_cbranch|s_branch)\s+(\.\w+|; %bb\.\d+)')

    current_label = None

    # Build the call tree from the lines to process
    for line in lines_to_process:
        line = line.strip()

        # Check for labels (blocks like .Ltmp, .LBB, and %bb)
        label_match = label_pattern.match(line)
        if label_match:
            current_label = label_match.group(1)
            call_tree.add_node(current_label)
            continue

        # Check for jump/branch instructions
        jump_match = jump_pattern.search(line)
        if jump_match and current_label:
            target_label = jump_match.group(2)
            call_tree.add_edge(current_label, target_label)

    block_registers = parse_register_usage(lines_to_process)

    for node in call_tree.nodes():
        if node.startswith('.LBB'):
            call_tree.nodes[node]['subset'] = 1
        elif node.startswith('.Ltmp'):
            call_tree.nodes[node]['subset'] = 2
        elif node.startswith('; %bb'):
            call_tree.nodes[node]['subset'] = 0
        else:
            call_tree.nodes[node]['subset'] = 3

    pos = nx.multipartite_layout(call_tree, subset_key="subset")

    plt.figure(figsize=(12, 10))

    lbb_nodes = [n for n in call_tree if n.startswith('.LBB')]
    ltmp_nodes = [n for n in call_tree if n.startswith('.Ltmp')]
    bb_nodes = [n for n in call_tree if n.startswith('; %bb')]
    other_nodes = [n for n in call_tree if n not in lbb_nodes and n not in ltmp_nodes and n not in bb_nodes]

    nx.draw_networkx_nodes(call_tree, pos, nodelist=lbb_nodes, node_color='skyblue', node_size=2000, label='.LBB Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=ltmp_nodes, node_color='lightgreen', node_size=1500, label='.Ltmp Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=bb_nodes, node_color='pink', node_size=1500, label='; %bb Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=other_nodes, node_color='orange', node_size=1500, label='Other Nodes')

    nx.draw_networkx_edges(call_tree, pos, arrows=True)

    labels = {
        node: f"{node}\nVGPRs: {len(block_registers[node]['vgprs'])}, SGPRs: {len(block_registers[node]['sgprs'])}"
        for node in call_tree.nodes()
    }
    nx.draw_networkx_labels(call_tree, pos, labels, font_size=8, font_weight='bold')

    plt.legend()
    plt.title("Call Tree and Register Usage (Per Block) for AMDGCN Assembly Blocks")
    plt.show()

    print("Register usage per block:")
    for block, usage in block_registers.items():
        print(f"Block {block}:")
        print(f"  VGPRs: {sorted(usage['vgprs'])}")
        print(f"  SGPRs: {sorted(usage['sgprs'])}")
        print(f"  VGPR Instruction Types:")
        for vgpr, types in usage['vgpr_instruction_types'].items():
            print(f"    v{vgpr}: {', '.join(types)}")

    blocks_in_call_sequence = get_calling_sequence(call_tree)

    plot_vgpr_usage(block_registers, blocks_in_call_sequence)

def get_calling_sequence(call_tree):
    starting_nodes = [n for n, d in call_tree.in_degree() if d == 0]

    if not starting_nodes:
        print("No starting nodes found in the call tree.")
        return []

    visited = set()
    calling_sequence = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        calling_sequence.append(node)
        for neighbor in call_tree.successors(node):
            dfs(neighbor)

    for start_node in starting_nodes:
        dfs(start_node)

    return calling_sequence

def plot_vgpr_usage(block_registers, blocks_in_call_sequence):
    blocks = blocks_in_call_sequence
    block_indices = {block: idx for idx, block in enumerate(blocks)}

    # Define colors for instruction types with priority
    instruction_colors = {
        'scratch_store': 'red',
        'scratch_load': 'green',
        'v_mfma': 'orange',
        'global_store': 'purple',
        'global_load': 'cyan',
        'others': 'blue',
    }

    # Define priority for instruction types
    instruction_priority = ['scratch_store', 'scratch_load', 'v_mfma', 'global_store', 'global_load', 'others']

    # Prepare data for plotting
    x_vals = []
    y_vals = []
    colors = []

    for block in blocks:
        idx = block_indices[block]
        vgprs = sorted(block_registers[block]['vgprs'])
        for vgpr in vgprs:
            x_vals.append(idx)
            y_vals.append(vgpr)
            instruction_types = block_registers[block]['vgpr_instruction_types'][vgpr]
            # Determine color based on highest priority instruction type
            color = 'blue'  # Default color
            for instr_type in instruction_priority:
                if instr_type in instruction_types:
                    color = instruction_colors[instr_type]
                    break
            colors.append(color)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, marker='s', s=20, c=colors)
    plt.xlabel('Block Index (Calling Sequence)')
    plt.ylabel('VGPR Register Index')
    plt.title('VGPR Usage per Block (Ordered by Calling Sequence)')
    plt.xticks(range(len(blocks)), blocks, rotation=90)
    plt.yticks(range(0, max(y_vals)+1, 8))  # Adjust the step as needed
    plt.grid(True)
    plt.tight_layout()

    # Add a legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color, label=instr_type.replace('_', ' ').title())
                      for instr_type, color in instruction_colors.items()]
    plt.legend(handles=legend_patches)

    plt.show()

# Call this function with the path to your assembly file
#generate_call_tree_and_register_usage('streamk_gemm.amdgcn')
generate_call_tree_and_register_usage('streamk_gemm_maskstore.amdgcn')
#generate_call_tree_and_register_usage('matmul_kernel_BM256_BN256_BK64_GM1_SK1_nW8_nS2_EU0_kP2_mfma16.amdgcn')

