import os.path


class StorageVariableData:

    def __init__(self, variable, file_name):
        self.write_file = None
        self.variable = variable
        self.file_name = file_name

    def open_write_variable_data(self):
        self.write_file = open(str(self.file_name), "w")
        try:
            print("Writing " + str(self.variable) + " data...")
            self.write_file.write(str(self.variable))
        except ValueError as write_error:
            print(write_error)

    def close_file(self):
        self.write_file.close()
        print("Finish writing into " + self.file_name)

    def print(self):
        print("class self var:\n", self.variable)
        print("class self file:\n", self.file_name)
