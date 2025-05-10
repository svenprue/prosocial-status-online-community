import os


def print_folder_structure(folder_path, prefix="", exclude_dirs={".venv", ".temp"}):
    """
    Print the complete folder structure excluding specified directories.

    Args:
        folder_path (str): The root folder path to traverse
        prefix (str): Prefix for indentation (used for recursion)
        exclude_dirs (set): Set of directory names to exclude
    """
    try:
        # Get all items in the current directory
        items = sorted(os.listdir(folder_path))

        # Separate directories and files
        dirs = []
        files = []

        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                if item not in exclude_dirs:  # Exclude unwanted directories
                    dirs.append(item)
            else:
                files.append(item)

        # Print directories first
        for i, dir_name in enumerate(dirs):
            dir_path = os.path.join(folder_path, dir_name)
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0

            # Print directory name
            print(f"{prefix}{'└── ' if is_last_dir else '├── '}{dir_name}/")

            # Recursively print subdirectories
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            print_folder_structure(dir_path, new_prefix, exclude_dirs)

        # Print files
        for i, file_name in enumerate(files):
            is_last_file = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last_file else '├── '}{file_name}")

    except PermissionError:
        print(f"{prefix}[Permission Denied]")
    except Exception as e:
        print(f"{prefix}[Error: {str(e)}]")


def main():
    # Get folder path from user or use current directory
    folder_path = input("Enter folder path (or press Enter for current directory): ").strip()

    if not folder_path:
        folder_path = os.getcwd()

    # Verify the path exists
    if not os.path.exists(folder_path):
        print(f"Error: Path '{folder_path}' does not exist.")
        return

    print(f"\nFolder structure for: {os.path.abspath(folder_path)}")
    print(f"(Excluding: .venv, .temp)")
    print()

    # Print the folder name as root
    print(os.path.basename(folder_path) or folder_path)

    # Start printing the structure
    print_folder_structure(folder_path)


if __name__ == "__main__":
    main()