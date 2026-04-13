"""
Standalone test runner with colored output for Heat_Flux_Diagnostic.
"""

from config import ModelConfig
from tests import run_basic_checks


# ANSI color codes (no external dependency)
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def main():
    print(f"{Colors.HEADER}{Colors.BOLD}Heat Flux Diagnostic - Test Runner{Colors.END}\n")

    try:
        print(f"{Colors.OKBLUE}Initializing configuration...{Colors.END}")
        config = ModelConfig()

        print(f"{Colors.OKCYAN}Running basic validation checks...{Colors.END}")
        run_basic_checks(config)

    except AssertionError as e:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Test Failed ❌{Colors.END}")
        print(f"{Colors.FAIL}{str(e)}{Colors.END}")
        return

    except Exception as e:
        print(f"\n{Colors.WARNING}{Colors.BOLD}Unexpected Error ⚠️{Colors.END}")
        print(f"{Colors.WARNING}{str(e)}{Colors.END}")
        return

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}All tests passed successfully ✅{Colors.END}")


if __name__ == "__main__":
    main()
