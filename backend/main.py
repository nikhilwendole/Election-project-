"""
main.py — Election Face Detection System
Uses OpenCV only — No dlib, No CMake!
"""

import sys
from database import init_db, save_voter, list_voters
from face_utils import live_face_verify


def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║       ELECTION FACE DETECTION SYSTEM             ║
║   MongoDB | OpenCV | No Duplicate Votes          ║
╚══════════════════════════════════════════════════╝
    """)


def print_menu():
    print("""
─────────────────────────────────
  MENU
  [1]  Cast Vote  (Face Verify)
  [2]  List All Voters
  [3]  Exit
─────────────────────────────────""")


def cast_vote():
    print("\n── CAST VOTE ──────────────────────────────")
    print("[INFO] Camera opening — look straight at camera")
    print("       GREEN = New Voter | RED = Already Voted\n")

    result_type, result_data = live_face_verify()

    if result_type == "cancelled":
        print("\n[INFO] Voting cancelled.")
        return

    if result_type == "duplicate":
        print("\n" + "="*50)
        print("  VOTE REJECTED — DUPLICATE VOTER!")
        print("="*50)
        print(f"  Name     : {result_data['name']}")
        print(f"  Voter ID : {result_data['voter_id']}")
        print(f"  Voted At : {result_data['voted_at']}")
        print("  This person has ALREADY voted!")
        print("="*50 + "\n")
        return

    if result_type == "new":
        encoding = result_data
        print("\n" + "="*50)
        print("  NEW VOTER VERIFIED!")
        print("="*50)
        name         = input("  Enter voter name      : ").strip()
        voter_number = input("  Enter voter number/ID : ").strip()

        if not name:
            print("[ERROR] Name required. Vote cancelled.")
            return

        doc = save_voter(name, encoding, voter_number)

        print("\n" + "="*50)
        print("  VOTE CAST SUCCESSFULLY!")
        print("="*50)
        print(f"  Name      : {doc['voter_name']}")
        print(f"  Voter No  : {doc['voter_number']}")
        print(f"  Voter ID  : {doc['voter_id']}")
        print(f"  Recorded  : {doc['voted_at']}")
        print("="*50 + "\n")


def main():
    print_banner()
    try:
        init_db()
    except Exception:
        print("\n[FATAL] Cannot connect to MongoDB!")
        print("  Make sure MongoDB is running.")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            cast_vote()
        elif choice == '2':
            list_voters()
        elif choice == '3':
            print("\n[EXIT] Goodbye!\n")
            sys.exit(0)
        else:
            print("[ERROR] Invalid choice. Enter 1, 2 or 3.")


if __name__ == "__main__":
    main()