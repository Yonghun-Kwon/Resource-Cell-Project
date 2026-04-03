# main.py
import argparse
from client import run_client
from server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["client", "server"], required=True)
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--client_id", type=str, help="Unique client id (for client mode)")
    parser.add_argument("--load_size", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "client":
        if not args.client_id:
            raise ValueError("--client_id is required for client mode")
        run_client(args.broker, args.client_id)
    else:
        run_server(args.broker)
