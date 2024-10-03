import subprocess
import sys
import io
import contextlib
import importlib
import os
import shutil

class PythonSandbox:
    def __init__(self):
        self.sandbox_dir = "/tmp/sandbox"
        self.packages = ["pandas", "matplotlib", "seaborn", "requests", "numpy"]
        
        self.create_sandbox()

    def create_sandbox(self):
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)
        os.makedirs(self.sandbox_dir)

        # venv_path = os.path.join(self.sandbox_dir, "venv")
        # subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

        
        # activate_script = os.path.join(venv_path, "bin", "activate")
        # activate_cmd = [".", activate_script]

        for package in self.packages:
            # subprocess.run(activate_cmd + ["&&", "pip", "install", package], shell=True, check=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def get_safe_modules(self):
        safe_modules = {
            "__builtins__": __builtins__,
            "os": os
        }
        for package in self.packages:
            safe_modules[package] = importlib.import_module(package)
        return safe_modules
    
    def cleanup_sandbox(self):
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)
    
    def run(self, args):
        code = args["code"]

        output = io.StringIO()
        error = io.StringIO()

        original_cwd = os.getcwd()
        try:
            os.chdir(self.sandbox_dir)

            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                try:
                    exec(code, self.get_safe_modules())
                except Exception as e:
                    pass

            output_str = output.getvalue()
            created_files = os.listdir(self.sandbox_dir)
            if created_files:
                output_str += f"\nCreated files: {', '.join(created_files)}"

                for file in created_files:
                    file_path = os.path.join(self.sandbox_dir, file)
                    output_str += f"\nFile '{file}' contents:\n"

                    if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        output_str += "[Binary image file]"
                    else:
                        try:
                            with open(file_path, 'r') as f:
                                output_str += f.read()
                        except Exception as e:
                            output_str += f"[Binary file]"
                
            output_str += f"\n\nError:\n{error.getvalue()}"
            self.cleanup_sandbox()
            os.chdir(original_cwd)
            return output_str
        
        except Exception as e:
            output_str = f"Error: {str(e)}"
            self.cleanup_sandbox()
            os.chdir(original_cwd)
            return output_str
