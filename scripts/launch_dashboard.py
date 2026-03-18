import subprocess


def main():
    try:
        print("Launching dashboard...")
        subprocess.run(["python", "dashboards/app.py"], check=True)
    except Exception as e:
        print("\n DASHBOARD LAUNCH FAILED")
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()