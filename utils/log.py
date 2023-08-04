class Logger:
    """
    Multiple output stream logger class.
    """


    def __init__(self, out_streams):
        """
        :param out_streams: list
            List of output streams.
            List may contain filenames, and if so, for each filename, the corresponging file will
            be opened in "write" mode and used as output stream.
        """

        self._initialize_streams(out_streams)


    def _initialize_streams(self, out_streams):
        """
        Initializes and validates output streams.
        Accepts:
          - Opened files
          - Filenames
        """
        
        self.out_streams = []

        for out_stream in out_streams:

            if type(out_stream) == str:
                
                out_stream = open(out_stream, "a+")
                self.out_streams.append(out_stream)
                
            else:

                self.out_streams.append(out_stream)


    def print(self, *args, **kwargs):
        """
        Logs a message to all output streams.
        Passes all arguments to the original "print" method.
        """

        for out_stream in self.out_streams:

            kwargs["file"] = out_stream
            print(*args, **kwargs)
            out_stream.flush()


    def flush(self):
        """
        Flushes out all out streams.
        """
        
        for out_stream in self.out_streams:

            out_stream.flush()
            