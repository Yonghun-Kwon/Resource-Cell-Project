import argparse
from client import send_inference_result
from server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['client', 'server'], required=True)
    args = parser.parse_args()

    if args.mode == 'client':
        send_inference_result("localhost", 1883)
    else:
        run_server()
