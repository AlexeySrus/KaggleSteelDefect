from visdom import Visdom
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Visdom')
    parser.add_argument('--server', required=False, type=str,
                          default='https://localhost',
                          help='Server address.'
                        )
    parser.add_argument('--port', required=False, type=str,
                            default='8080',
                            help='Port.'
                        )
    parser.add_argument('--logfile', required=True, type=str,
                            help='Path to logfile.'
                        )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    viz = Visdom(server=args.server, port=args.port)
    viz.replay_log(args.logfile)
