"""Settings class, subclass of dictionary."""

class Settings(dict):
    """ Handler class to user defined parameters.

        Usage:
            options = Settings.read(settings_filename)
        
        Keyword arguments are written in settings_filename.
        Settings.read() returns them as key:value pairs in options.
    """

    def __init__(self, **kwargs):
        """Assign values to keys from options file."""
        for key, value in kwargs.items():
            self[key] = value

    @classmethod
    def read(cls, filename):
        """Load user parameters from options file."""
        options = {}
        # Load user parameters
        with open(filename, "r") as f:
            f = "\n".join(f.readlines())
            exec(f, options)

        return cls(**options)
