# Specify the path to the file (including the filename and extension)
import importlib.util
import sys

class ModuleLoader:
    def __init__(self, module_name, module_path):
        self.module_name = module_name
        self.module_path = module_path

    def load_module(self):
        """Load a module from a specified file location and add it to sys.modules."""
        if self.module_name in sys.modules:
            # Return the module if it's already loaded
            return sys.modules[self.module_name]
        
        # Define the module specification from the file location
        spec = importlib.util.spec_from_file_location(self.module_name, self.module_path)
        if spec and spec.loader:
            # Create and load the module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[self.module_name] = module
            print(f"Module {self.module_name} loaded successfully from {self.module_path}")
            return module
        else:
            raise ImportError(f"Could not load the module {self.module_name} from {self.module_path}")