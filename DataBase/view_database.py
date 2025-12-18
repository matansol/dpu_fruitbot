"""
Script to connect to Azure Database and display contents of dpu_fruitbot tables.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Get database URI from environment
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")

# Create engine and session
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

def get_table_names():
    """Get all table names from the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()

def display_table(table_name, limit=None):
    """Display contents of a specific table."""
    try:
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        print(f"\n{'='*80}")
        print(f"Table: {table_name}")
        print(f"Total rows: {len(df)}")
        print(f"{'='*80}")
        
        if len(df) == 0:
            print("  (No data in this table)")
        else:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        return df
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
        return None

def display_table_summary(table_name):
    """Display summary statistics for a table."""
    try:
        query = f"SELECT * FROM {table_name}"
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        print(f"\n{'='*80}")
        print(f"Summary for table: {table_name}")
        print(f"{'='*80}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        if len(df) > 0:
            print("\nColumn Info:")
            print(df.dtypes.to_string())
            
            print("\nFirst 5 rows:")
            print(tabulate(df.head(), headers='keys', tablefmt='grid', showindex=False))
        else:
            print("  (No data in this table)")
        
        return df
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
        return None

def main():
    """Main function to display database contents."""
    print(f"Connecting to database...")
    print(f"Database URI: {DATABASE_URI[:50]}..." if len(DATABASE_URI) > 50 else f"Database URI: {DATABASE_URI}")
    
    # Get all tables
    tables = get_table_names()
    print(f"\nFound {len(tables)} tables: {', '.join(tables)}")
    
    # Display each table
    for table in tables:
        display_table_summary(table)
    
    # Interactive mode
    print("\n" + "="*80)
    print("Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Type table name to view full contents")
    print("  - Type 'summary <table_name>' for summary")
    print("  - Type 'export <table_name> <filename.csv>' to export to CSV")
    print("  - Type 'list' to list all tables")
    print("  - Type 'quit' to exit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'list':
                print(f"Tables: {', '.join(tables)}")
            elif command.lower().startswith('summary '):
                table_name = command.split()[1]
                if table_name in tables:
                    display_table_summary(table_name)
                else:
                    print(f"Table '{table_name}' not found")
            elif command.lower().startswith('export '):
                parts = command.split()
                if len(parts) == 3:
                    table_name = parts[1]
                    filename = parts[2]
                    if table_name in tables:
                        with engine.connect() as conn:
                            df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)
                        df.to_csv(filename, index=False)
                        print(f"Exported {len(df)} rows to {filename}")
                    else:
                        print(f"Table '{table_name}' not found")
                else:
                    print("Usage: export <table_name> <filename.csv>")
            elif command in tables:
                display_table(command)
            else:
                print(f"Unknown command or table: {command}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
