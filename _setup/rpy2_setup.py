# Import necessary libraries for OS operations and R integration
import os
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks as callbacks


def custom_write_console(output):
    """
    Custom function to handle output from R to the Python console.

    Args:
    output (str): The output string from R.

    This function attempts to print output directly. If an error occurs
    (e.g., output not being a proper string), it catches the exception and prints an error message.
    Additionally, it prints a generic warning message about potential decoding issues.
    """
    try:
        # Attempt to print the R output directly
        print(output)
    except Exception as e:
        # Print any exception that occurs during the printing
        print("Error:", e)

    # Print a static warning about potential string decoding issues
    print("Warning: 'str' object has no attribute 'decode'")


def setup_environment():
    """
    Set up the environment for R integration with Python using rpy2.

    This function sets the R_HOME environment variable, configures how Python should handle
    R outputs via custom callback functions, and loads necessary R libraries.
    """
    # Specify the path to the R installation directory
    os.environ['R_HOME'] = 'C:/Users/MatarKANDJI/R/R-4.3.3'

    # Configure rpy2 callback functions to handle R console outputs
    callbacks.consolewrite_print = custom_write_console
    callbacks.consolewrite_warnerror = custom_write_console

    # Execute R commands to set locale and load required R packages
    ro.r('''
        Sys.setlocale("LC_ALL", "C")        
        options(repos = 'http://cran.rstudio.com/')
        library(rugarch)
        library(rmgarch)
        library(parallel)
        ''')

    # Print a confirmation once the environment is set up successfully
    print("Environment setup complete.")


# Check if the script is the main program and if so, call the setup_environment function
if __name__ == "__main__":
    setup_environment()
