DEBUG_MODE = True

class Debugger:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    @staticmethod
    def log(message):
        if DEBUG_MODE:
            print(f"{Debugger.GREEN}[DEBUG]{Debugger.RESET} {message}")

    @staticmethod
    def warn(message):
        if DEBUG_MODE:
            print(f"{Debugger.YELLOW}[WARN]{Debugger.RESET} {message}")

    @staticmethod
    def error(message):
        if DEBUG_MODE:
            print(f"{Debugger.RED}[ERROR]{Debugger.RESET} {message}")
