"""
Delete specific rows from user_choices by (user_id, episode_index) pairs.

Usage:
    python DataBase/remove_user_choices_by_pairs.py

Optional dry run:
    python DataBase/remove_user_choices_by_pairs.py --dry-run
"""
import os
import argparse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")

# Pairs to remove from user_choices table.
CHOICES_TO_REMOVE = [
    ("69b04ae112b55b3a183c8ea9", 2),
    ("698c054ad44c45a1061327b9", 3),
    ("698c054ad44c45a1061327b9", 5),
    ("673f6ffebce20832346325de", 1),
    ("69ac96e9cf57af07fdd54227", 4),
    ("5805db219ff6120001a135a2", 1),
    ("69832ecb3693ee29d4fcc3c6", 2),
    ("6765ef2feed02bfb60182ecb", 4),
    ("69c429803583836ef035bd0d", 1),
    ("66b1375b72c3a0f57828dcca", 3),
    ("6913b066f2c9da9a92afc734", 3),
    ("69ac9523ad63eebc23c358ec", 4),
    ("65313bdcb9fd825a8c1d4981", 4),
    ("678b6eba161e3e871bbafef5", 3),
    ('6651e7f3cec5c25f42720bfa', 1)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete rows from user_choices by pair list")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show how many rows would be deleted, without deleting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(DATABASE_URI)

    print("Connecting to database...")
    print(f"Using URI: {DATABASE_URI[:50]}..." if len(DATABASE_URI) > 50 else f"Using URI: {DATABASE_URI}")

    count_stmt = text(
        """
        SELECT COUNT(*)
        FROM user_choices
        WHERE user_id = :user_id AND episode_index = :episode_index
        """
    )

    delete_stmt = text(
        """
        DELETE FROM user_choices
        WHERE user_id = :user_id AND episode_index = :episode_index
        """
    )

    total_found = 0
    total_deleted = 0

    try:
        with engine.begin() as conn:
            for user_id, episode_index in CHOICES_TO_REMOVE:
                found = conn.execute(
                    count_stmt,
                    {"user_id": user_id, "episode_index": episode_index},
                ).scalar_one()

                if found:
                    total_found += found
                    print(f"Found {found} row(s) for ({user_id}, {episode_index})")

                    if not args.dry_run:
                        result = conn.execute(
                            delete_stmt,
                            {"user_id": user_id, "episode_index": episode_index},
                        )
                        total_deleted += result.rowcount
                        print(f"Deleted {result.rowcount} row(s) for ({user_id}, {episode_index})")
                else:
                    print(f"No rows found for ({user_id}, {episode_index})")

            if args.dry_run:
                print("\nDry run complete: no changes committed.")
                # engine.begin() would auto-commit on success, so rollback for dry-run.
                conn.rollback()

        print("\nSummary")
        print(f"Pairs checked: {len(CHOICES_TO_REMOVE)}")
        print(f"Rows found: {total_found}")
        if args.dry_run:
            print("Rows deleted: 0 (dry run)")
        else:
            print(f"Rows deleted: {total_deleted}")
            print("Done.")

    except Exception as exc:
        print(f"Error while removing rows: {exc}")


if __name__ == "__main__":
    main()
