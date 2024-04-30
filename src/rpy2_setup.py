import os
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks as callbacks

def custom_write_console(output):
    try:
        # Print the output directly, assuming it's already a string
        print(output)
    except Exception as e:
        print("Error:", e)

    # Print the warning message
    print("Warning: 'str' object has no attribute 'decode'")


def setup_environment():
    # Set R_HOME to the R installation directory if needed
    os.environ['R_HOME'] = 'C:/Users/MatarKANDJI/R/R-4.3.3'

    # Configure rpy2 to handle R outputs using the custom function
    callbacks.consolewrite_print = custom_write_console
    callbacks.consolewrite_warnerror = custom_write_console

    # Activate automatic conversion of pandas dataframes to R data.frames
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    # Ensure R uses the C locale to prevent locale-specific issues
    ro.r('Sys.setlocale("LC_ALL", "C")')

    print("Environment setup complete.")


if __name__ == "__main__":
    setup_environment()
