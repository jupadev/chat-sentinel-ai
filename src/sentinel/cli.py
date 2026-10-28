"""Admin commands: clients and API keys.

    python -m sentinel.cli create-client <name> [--webhook-url URL]
    python -m sentinel.cli create-key <client-name> [--scopes messages:write,moderation:read,review:write]
    python -m sentinel.cli list-keys
    python -m sentinel.cli revoke-key <key-id>
"""

import argparse
import sys

from sentinel.api import SCOPES
from sentinel.config import get_config
from sentinel.store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel")
    sub = parser.add_subparsers(dest="command", required=True)

    create_client = sub.add_parser("create-client")
    create_client.add_argument("name")
    create_client.add_argument("--webhook-url")

    create_key = sub.add_parser("create-key")
    create_key.add_argument("client")
    create_key.add_argument("--scopes", default=",".join(sorted(SCOPES)))

    sub.add_parser("list-keys")

    revoke = sub.add_parser("revoke-key")
    revoke.add_argument("key_id")

    args = parser.parse_args(argv)
    store = Store(get_config().database_path)

    if args.command == "create-client":
        client_id, secret = store.create_client(args.name, args.webhook_url)
        print(f"client_id:      {client_id}")
        print(f"webhook_secret: {secret}   (verify X-Sentinel-Signature with it; shown once)")
    elif args.command == "create-key":
        client = store.client_by_name(args.client)
        if client is None:
            print(f"unknown client {args.client}", file=sys.stderr)
            return 1
        scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
        if unknown := set(scopes) - SCOPES:
            print(f"unknown scopes {sorted(unknown)}; valid: {sorted(SCOPES)}", file=sys.stderr)
            return 1
        key_id, raw = store.create_api_key(client["id"], scopes)
        print(f"key_id:  {key_id}")
        print(f"api_key: {raw}   (shown once; only its hash is stored)")
    elif args.command == "list-keys":
        for row in store.list_api_keys():
            state = f"revoked {row['revoked_at']}" if row["revoked_at"] else "active"
            print(f"{row['id']}  {row['client']:<20} {row['prefix']}…  {row['scopes']}  {state}")
    elif args.command == "revoke-key":
        if not store.revoke_api_key(args.key_id):
            print("no active key with that id", file=sys.stderr)
            return 1
        print("revoked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
