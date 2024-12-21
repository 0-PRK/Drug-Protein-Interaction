import yaml
import os


# Load the environment.yml file
with open("environment.yml", "r") as file:
    env = yaml.safe_load(file)

# Extract pip dependencies
pip_deps = env.get("dependencies", [])
for dep in pip_deps:
    if isinstance(dep, dict) and "pip" in dep:
        # Correct the output path
        output_path = r"C:\My Files\KU\6th sem\Bio-Hakathon\Drug-Protein-Interaction\backend\requirements.txt"
        print(f"Writing requirements.txt to: {os.path.abspath(output_path)}")
        with open(output_path, "w") as req_file:
            req_file.write("\n".join(dep["pip"]))
        break

print("Requirements.txt has been created!")
