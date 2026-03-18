from src.pipeline.config import build_config
from src.pipeline.run_pipeline import run_full_pipeline

import subprocess


def get_user_input():
    print("\n=== Diffusion Topic Pipeline ===")

    try:
        start_year = int(input("Enter start year: ").strip())
        end_year = int(input("Enter end year: ").strip())

    except ValueError:
        raise ValueError("Years must be integers (e.g., 2020)")

    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year")

    return start_year, end_year


def launch_dashboard():
    print("\n🚀 Launching dashboard...")
    try:
        subprocess.run(["python", "dashboards/app.py"], check=True)
    except Exception as e:
        print("\n Failed to launch dashboard")
        print(str(e))


def main():
    try:
        start_year, end_year = get_user_input()
        config = build_config(start_year, end_year)

        success = run_full_pipeline(config)

        if success:
            launch_dashboard()
        else:
            print("\n Pipeline failed — dashboard will not launch.")

    except Exception as e:
        print("\n INPUT ERROR")
        print(str(e))


if __name__ == "__main__":
    main()