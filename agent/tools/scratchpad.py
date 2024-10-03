class Scratchpad:
    def __init__(self):
        self.scratchpad = ""

    def write(self, args):
        self.scratchpad += args["content"]
        return "Successfully wrote to scratchpad"

    def read(self, args):
        return self.scratchpad
    