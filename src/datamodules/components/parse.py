import os


def parse_image_paths(
    file_paths: list[str] | None = None,
    dir_paths: list[str] | None = None,
    txt_paths: list[str] | None = None,
    xts: tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
) -> list[str]:
    """Parse image paths from List, List of directories or List of txt files.

    :param file_paths: List of files.
    :param dir_paths: List of directories.
    :param txt_paths: List of TXT files.
    :param xts: Image extensions which are allowed.
        Default to ('.jpg', '.jpeg', '.png').
    :return: Parsed images paths.
    """

    paths = []
    if file_paths:
        for file_path in file_paths:
            _, xt = os.path.splitext(file_path)
            if xt in xts:
                paths.append(file_path)

    if dir_paths:
        for dir_path in dir_paths:
            for d, dirs, files in os.walk(dir_path):
                for file in files:
                    _, xt = os.path.splitext(file)
                    if xt in xts:
                        paths.append(os.path.join(d, file))

    if txt_paths:
        for txt_path in txt_paths:
            with open(txt_path) as txt_file:
                for line in txt_file:
                    path = line[:-1].split('\t')[-1]
                    paths.append(path)

    if not paths:
        raise ValueError(f'No images with extensions: {xts}')
    return paths
