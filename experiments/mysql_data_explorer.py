import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MySQLDataExplorer:
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 3306):
        """
        Initialize MySQL connection parameters for Azure MySQL
        
        Args:
            host: Azure MySQL server hostname
            database: Database name
            user: Username
            password: Password
            port: Port number (default: 3306)
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        
    def connect(self) -> bool:
        """
        Establish connection to Azure MySQL database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                ssl_disabled=False,
                autocommit=True
            )
            
            if self.connection.is_connected():
                print(f"✅ Connected to Azure MySQL database: {self.database}")
                return True
                
        except Error as e:
            print(f"❌ Error connecting to Azure MySQL: {e}")
            return False
        return False
    
    def get_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Get all tables from the database as pandas DataFrames
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with table names as keys and DataFrames as values
        """
        if not self.connection or not self.connection.is_connected():
            print("❌ Not connected to database")
            return {}
            
        tables_dict = {}
        
        try:
            # Get list of all tables
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
            
            print(f"📊 Found {len(tables)} tables: {tables}")
            
            # Load each table into a DataFrame
            for table_name in tables:
                try:
                    query = f"SELECT * FROM {table_name}"
                    df = pd.read_sql(query, self.connection)
                    tables_dict[table_name] = df
                    print(f"   ✅ Loaded '{table_name}': {len(df)} rows, {len(df.columns)} columns")
                except Error as e:
                    print(f"   ❌ Error loading table '{table_name}': {e}")
            
            return tables_dict
            
        except Error as e:
            print(f"❌ Error getting tables: {e}")
            return {}
    
    def close_connection(self):
        """Close the database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Database connection closed")


def get_all_tables_from_mysql() -> Dict[str, pd.DataFrame]:
    """
    Simple function to connect to MySQL and get all tables
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with table names as keys and DataFrames as values
    """
    # Load configuration from .env file
    config = {
        'host': os.getenv('AZURE_MYSQL_HOST'),
        'database': os.getenv('AZURE_MYSQL_DATABASE'),
        'user': os.getenv('AZURE_MYSQL_USER'),
        'password': os.getenv('AZURE_MYSQL_PASSWORD'),
        'port': int(os.getenv('AZURE_MYSQL_PORT', 3306))
    }
    
    # Check if all required environment variables are set
    missing_vars = [key for key, value in config.items() if not value and key != 'port']
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return {}
    
    # Create explorer instance and connect
    explorer = MySQLDataExplorer(**config)
    
    if explorer.connect():
        try:
            # Get all tables
            tables_dict = explorer.get_all_tables()
            return tables_dict
        finally:
            explorer.close_connection()
    else:
        print("❌ Failed to connect to database")
        return {}

def alter_database_tables(query: str):
    """
    Alters the 'user_choices' table by adding new columns.
    """
    # Load configuration from .env file
    config = {
        'host': os.getenv('AZURE_MYSQL_HOST'),
        'database': os.getenv('AZURE_MYSQL_DATABASE'),
        'user': os.getenv('AZURE_MYSQL_USER'),
        'password': os.getenv('AZURE_MYSQL_PASSWORD'),
        'port': int(os.getenv('AZURE_MYSQL_PORT', 3306))
    }

    explorer = MySQLDataExplorer(**config)
    if explorer.connect():
        try:
            cursor = explorer.connection.cursor()
            # Add unique_env column
            cursor.execute(query)
            print("✅ ", query)
            cursor.close()
        except Error as e:
            print(f"❌ Error altering table: {e}")
        finally:
            explorer.close_connection()
    else:
        print("❌ Failed to connect to database")


def main():
    """
    Example usage
    """
    tables_dict = get_all_tables_from_mysql()
    
    if tables_dict:
        print(f"\n🎯 Successfully loaded {len(tables_dict)} tables:")
        for table_name, df in tables_dict.items():
            print(f"   📊 {table_name}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Example: Access specific tables
        if 'actions' in tables_dict:
            print(f"\n🎮 Actions table preview:")
            print(tables_dict['actions'].head())
        
        if 'feedback_actions' in tables_dict:
            print(f"\n💬 Feedback actions table preview:")
            print(tables_dict['feedback_actions'].head())
        
        if 'user_choices' in tables_dict:
            print(f"\n🎯 User choices table preview:")
            print(tables_dict['user_choices'].head())
    else:
        print("❌ No tables loaded")


if __name__ == "__main__":
    main() 