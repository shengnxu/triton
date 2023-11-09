import argparse
import os
import tempfile
from git import Repo, exc

def clone_repo(repo_url, local_path):
    print(f"Cloning {repo_url} into {local_path}")
    Repo.clone_from(repo_url, local_path)

def find_last_shared_commit(upstream_repo, fork_repo):
    fork_commits = set(fork_repo.git.rev_list('origin/' + fork_repo.active_branch.name).split())
    upstream_commits = upstream_repo.git.rev_list('origin/' + upstream_repo.active_branch.name).split()

    for commit in upstream_commits:
        if commit in fork_commits:
            return commit
    return None

def checkout_commit(repo, commit):
    repo.git.checkout(commit)

def show_diff(upstream, fork, path):
    if path:
        return fork.git.diff(upstream.head.commit, path)
    else:
        return fork.git.diff(upstream.head.commit)

def get_repo_name_from_url(repo_url):
    return repo_url.split('/')[-1].replace('.git', '')

def show_commit_message(repo, commit):
    return repo.git.log(commit, '-1', '--pretty=%B').strip()

def main(upstream_repo_url, fork_repo_url, path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        base_repo_name = get_repo_name_from_url(upstream_repo_url)
        upstream_repo_path = os.path.join(tmpdirname, f"{base_repo_name}_upstream")
        fork_repo_path = os.path.join(tmpdirname, f"{base_repo_name}_fork")

        try:
            # Clone repositories
            clone_repo(upstream_repo_url, upstream_repo_path)
            clone_repo(fork_repo_url, fork_repo_path)

            upstream_repo = Repo(upstream_repo_path)
            fork_repo = Repo(fork_repo_path)

            # Ensure the repositories are on their default branches
            upstream_repo.git.checkout(upstream_repo.active_branch)
            fork_repo.git.checkout(fork_repo.active_branch)

            # Find the last shared commit
            last_shared_commit = find_last_shared_commit(upstream_repo, fork_repo)
            if last_shared_commit:
                print(f"SHARED_COMMIT_HASH={last_shared_commit}")
                print(f"SHARED_COMMIT_URL={upstream_repo_url}/commit/{last_shared_commit}")

                # Show the log for the shared commit with tab indentation for each line
                commit_log = show_commit_message(upstream_repo, last_shared_commit)
                indented_log = '\n\t'.join(commit_log.split('\n'))
                print(f"Log for the shared commit:\n\t{indented_log}")

                # Checkout the upstream repo at that commit
                checkout_commit(upstream_repo, last_shared_commit)

                # Get the diff
                diff_output = show_diff(upstream_repo, fork_repo, path)

                # Write the diff to a file
                if path:
                    diff_file_name = f"{base_repo_name}_{path.replace('/','_')}.diff"
                else:
                    diff_file_name = f"{base_repo_name}.diff"
                with open(diff_file_name, 'w') as file:
                    file.write(diff_output)
                print(f"Diff written to {diff_file_name}")
            else:
                print("No common commit found between the default branches of the upstream and fork repositories.")
        except exc.InvalidGitRepositoryError:
            print("Invalid Git repository path provided.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the last shared commit between the default branches of an upstream repository and a fork, checkout the upstream at that commit, and show the diff.")
    parser.add_argument("--upstream", required=True, help="URL of the upstream GitHub repository")
    parser.add_argument("--fork", required=True, help="URL of the fork GitHub repository")
    parser.add_argument("--path", default='', help="Optional path to a specific file or folder")

    args = parser.parse_args()
    main(args.upstream, args.fork, args.path)