"""
Launcher script for MPC trajectory tracking applications
"""
import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='MPC Trajectory Tracking Launcher')
    parser.add_argument('--interface', type=str, default='menu',
                        choices=['menu', 'cli', 'web'],
                        help='Interface type: menu, cli, web')
    
    args = parser.parse_args()
    
    if args.interface == 'menu':
        print("🚗 MPC Trajectory Tracking Launcher")
        print("=" * 50)
        print("Choose application:")
        print("1. Command Line Interface")
        print("2. Enhanced Web Interface (4-Point Cubic)")
        print("3. Exit")
        print("=" * 50)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🖥️  Starting Command Line Interface...")
            print("Usage examples:")
            print("  python main_simple.py --trajectory ref1")
            print("  python main_simple.py --trajectory ref2 --speed 5.0")
            print("  python main_simple.py --trajectory ref3 --horizon 80")
            print("\nRunning default (ref1)...")
            subprocess.run([sys.executable, "main_simple.py", "--trajectory", "ref1"])
            
        elif choice == "2":
            print("\n🌐 Starting Enhanced Web Interface...")
            print("This opens the enhanced interface with perfect 4-point cubic trajectories!")
            print("URL: http://localhost:8085")
            print("Features: 4-point cubic polynomial + Real MPC simulation + GIF animation!")
            subprocess.run([sys.executable, "web_interface.py"])
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            return
            
    elif args.interface == 'cli':
        print("\n🖥️  Starting Command Line Interface...")
        subprocess.run([sys.executable, "main_simple.py", "--trajectory", "ref1"])
        
    elif args.interface == 'enhanced':
        print("\n❌ Enhanced Gradio MPC Simulator is no longer available.")
        print("The enhanced interface has been integrated into the web interface.")
        print("Please use option 2 for the web interface instead.")
        print("URL: http://localhost:8085")
        return
        
    elif args.interface == 'main':
        print("\n🚀 Starting Main MPC Simulation...")
        subprocess.run([sys.executable, "main.py"])
    
    else:
        print(f"\n❌ Unknown interface: {args.interface}")
        print("Available: menu, cli, enhanced, main")

if __name__ == "__main__":
    main()
