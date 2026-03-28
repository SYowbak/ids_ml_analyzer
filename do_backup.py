import os
import subprocess

def run_backup():
    print("Running git add .")
    subprocess.run(["git", "add", "."], check=True)
    print("Running git commit")
    subprocess.run(["git", "commit", "-m", "Second backup"], check=False) # may fail if nothing to commit
    print("Running git push")
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Backup completed successfully.")

if __name__ == "__main__":
    run_backup()
