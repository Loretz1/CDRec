import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Base', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon2014', help='name of dataset')
    parser.add_argument('--domains', type=str, nargs='+', default='Clothing_Shoes_and_Jewelry Sports_and_Outdoors',
                        help='List of domains')

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, domains = args.domains, save_model=True)


