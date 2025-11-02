#!/usr/bin/env python3
"""
Demonstration of the Parameterized Function Plotting System
Shows how functions are now passed as parameters to the data generator
"""

import subprocess
import os

def run_lean_and_plot(lean_file, description):
    """Run a Lean file and then plot the results"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    # Run Lean data generation
    print("📊 Generating data with Lean...")
    try:
        result = subprocess.run(['lean', lean_file], 
                              capture_output=True, text=True, check=True)
        print("✓ Lean execution completed")
        if result.stdout.strip():
            # Print just the key info from Lean output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Function:' in line or 'Generated' in line or 'Range:' in line:
                    print(f"  {line}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False
    
    # Plot the results  
    print("\n📈 Creating plot...")
    try:
        result = subprocess.run(['python', 'auto_plot.py'], 
                              capture_output=True, text=True, check=True)
        print("✅ Plot generated successfully!")
        
        # Extract and show the function analysis
        lines = result.stdout.split('\n')
        in_analysis = False
        for line in lines:
            if '📊 Function Analysis:' in line:
                in_analysis = True
            elif in_analysis and line.strip():
                if line.startswith('   '):
                    print(line)
                elif '✅' in line:
                    break
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Plot error: {e.stderr}")
        return False

def demonstrate_parameterized_system():
    """Show off the parameterized function system"""
    print("🎯 PARAMETERIZED FUNCTION PLOTTING DEMONSTRATION")
    print("=" * 70)
    print("This demo shows how functions are now passed as parameters!")
    print()
    
    # Demo 1: Current quadratic function
    if run_lean_and_plot("DataGenerator.lean", "Demo 1: Parameterized Quadratic Function"):
        input("\nPress Enter to continue to the next demo...")
    
    # Demo 2: Multi-function example
    if run_lean_and_plot("MultiFunctionExample.lean", "Demo 2: Multi-Function Parameterized Example"):
        input("\nPress Enter to see the system overview...")
    
    # Show the power of the parameterized system
    print("\n" + "=" * 70)
    print("🎉 PARAMETERIZED SYSTEM BENEFITS")
    print("=" * 70)
    print("""
✨ What makes this system powerful:

1. 🔧 FUNCTIONS AS PARAMETERS
   - Functions are passed to plotFunction() as arguments
   - No need to edit the plotting infrastructure
   - Completely modular and reusable

2. ⚙️ CONFIGURABLE PARAMETERS  
   - Each function call can specify its own:
     • Domain range (start, stop)
     • Resolution (number of points)
     • Special points to highlight
     • Display name and expression

3. 📚 FUNCTION LIBRARIES
   - Define families of related functions
   - Parameterized functions (like quadratic(a,b,c))
   - Reusable function definitions

4. 🎯 TYPE SAFETY
   - Functions are properly typed as Float → Float
   - Lean's type system ensures correctness
   - No runtime function errors

EXAMPLE OF THE POWER:
Instead of editing code every time, you can now do:

    #eval plotFunction quadratic "f₁" "3x² - 4x + 5" (-2.0) 4.0 100 [0,1,3]
    #eval plotFunction sine "f₂" "sin(x)" (-6.28) 6.28 200 [0,π/2,π]
    #eval plotFunction myCustom "f₃" "x·e^(-x²)" (-3.0) 3.0 300 [-2,0,2]

All with the same plotting infrastructure! 🚀
""")

def show_file_overview():
    """Show what each file does in the new system"""
    print("\n" + "=" * 70)
    print("📁 FILE STRUCTURE OVERVIEW")
    print("=" * 70)
    
    files_info = {
        "DataGenerator.lean": "🔧 Main parameterized plotting system - EDIT THIS for new functions",
        "MultiFunctionExample.lean": "📚 Multiple function examples with different parameters",
        "FunctionTemplates.lean": "📝 Copy-paste templates for common functions",
        "SineExample.lean": "🌊 Pre-configured sine wave example",
        "auto_plot.py": "🎨 Enhanced Python plotter (handles any parameterized function)",
        "parameterized_plotter.py": "🚀 Interactive function selector with parameterized support",
        "README_Plotter.md": "📖 Complete documentation of the parameterized system"
    }
    
    for filename, description in files_info.items():
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"{status} {filename:<25} - {description}")
    
    print("""
🎯 TO GET STARTED:
1. Run: python parameterized_plotter.py
2. Or edit DataGenerator.lean and add your own function calls
3. Or explore MultiFunctionExample.lean for advanced examples

The parameterized system gives you the power of functional programming
with mathematical precision and beautiful visualizations! 🎉
""")

if __name__ == "__main__":
    demonstrate_parameterized_system()
    show_file_overview()