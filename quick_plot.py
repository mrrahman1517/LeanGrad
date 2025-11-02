#!/usr/bin/env python3
"""
Quick Function Plotter - Easy switching between different functions
"""

import shutil
import os
import subprocess

# Available function examples
EXAMPLES = {
    "1": {
        "name": "Quadratic (current)",
        "file": "DataGenerator.lean",
        "desc": "f(x) = 3x² - 4x + 5"
    },
    "2": {
        "name": "Sine wave",
        "file": "SineExample.lean", 
        "desc": "f(x) = sin(x)"
    },
    "3": {
        "name": "Custom",
        "file": None,
        "desc": "Edit DataGenerator.lean manually"
    }
}

def show_menu():
    print("🚀 Quick Function Plotter")
    print("=" * 40)
    print("Available functions:")
    for key, ex in EXAMPLES.items():
        print(f"  {key}. {ex['name']}: {ex['desc']}")
    print()

def plot_function(example_key):
    """Switch to a function example and plot it"""
    if example_key not in EXAMPLES:
        print(f"❌ Invalid choice: {example_key}")
        return False
        
    example = EXAMPLES[example_key]
    
    if example_key == "3":
        print("📝 Opening DataGenerator.lean for manual editing...")
        print("💡 Use the templates in FunctionTemplates.lean for inspiration!")
        return True
    
    if example["file"] and os.path.exists(example["file"]):
        if example_key != "1":  # Don't copy if already using DataGenerator.lean
            print(f"📋 Switching to: {example['name']}")
            shutil.copy(example["file"], "DataGenerator.lean")
        
        print(f"📊 Plotting: {example['desc']}")
        
        # Run the plotter
        try:
            result = subprocess.run(['python', 'auto_plot.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Plot generated successfully!")
                return True
            else:
                print(f"❌ Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error running plotter: {e}")
            return False
    else:
        print(f"❌ Example file not found: {example['file']}")
        return False

def main():
    show_menu()
    
    while True:
        choice = input("Choose a function to plot (1-3, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if choice in EXAMPLES:
            if plot_function(choice):
                print("\n" + "="*50)
                show_menu()
            else:
                print("❌ Failed to plot function. Please try again.")
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'.")

if __name__ == "__main__":
    main()