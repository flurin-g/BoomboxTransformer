import argparse
from utils import h_params, create_libri_meta, create_urban_meta


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('--task', type=str, choices=['train', 'create-meta'], required=True,
                        help='Choose task to run')

    return parser


def main():
    parser = argparse.ArgumentParser(description='PyTorch BoomboxTransformer')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    if "create-meta" in args.task:
        create_libri_meta(libri_path=h_params.libri_path,
                          libri_meta_path=h_params.libri_speakers,
                          file_name=h_params.libri_meta,
                          drop_subsets=h_params.libri_drop_subsets)

        create_urban_meta(urban_path=h_params.urban_path)


if __name__ == '__main__':
    main()
