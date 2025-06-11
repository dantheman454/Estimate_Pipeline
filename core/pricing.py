import sqlite3
import os

class PricingEngine:
    def __init__(self, db_path: str = 'pricing.db'):
        """Initialize pricing engine with SQLite database for electrical components.

        Creates or connects to SQLite database containing electrical component
        prices for residential estimation. Automatically initializes with
        standard residential electrical component pricing.

        Args:
            db_path (str): Path to SQLite database file. Defaults to 'pricing.db'
            in current directory. Uses ':memory:' for in-memory database.

        Returns:
            None: Constructor method.

        Algorithm:
            1. Store database path and establish SQLite connection
            2. Call _init_db() to create tables and populate with default prices
            3. Ready for price queries and updates

        Related Functions:
            _init_db: Creates database schema and populates default prices
            close: Properly closes database connection when done

        """
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Create database schema and populate with standard electrical component prices.

        Initializes SQLite database with 'prices' table for electrical components
        and populates with current market rates for residential electrical work.
        Uses INSERT OR REPLACE to handle existing data gracefully.

        Args:
            None: Internal method uses instance connection.

        Returns:
            None: Database initialization method.

        Algorithm:
            1. Create 'prices' table with component name and unit price columns
            2. Define standard electrical component prices (10 component types)
            3. Insert all prices using INSERT OR REPLACE for idempotency
            4. Commit transaction to persist changes

        Related Functions:
            __init__: Calls this method during initialization
            update_price: Individual price updates after initialization

        """
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                component TEXT PRIMARY KEY,
                unit_price REAL
            )
        ''')
        
        # Insert residential electrical component prices
        prices = [
            ('light_switch', 18.75),
            ('outlet', 25.50),
            ('ceiling_fan', 125.00),
            ('smoke_detector', 35.00),
            ('light_fixture', 45.00),
            ('junction_box', 12.30),
            ('electrical_panel', 275.00),
            ('doorbell', 45.00),
            ('thermostat', 120.00),
            ('gfci_outlet', 32.00)
        ]
        
        self.conn.executemany('INSERT OR REPLACE INTO prices VALUES (?, ?)', prices)
        self.conn.commit()
    
    def get_price(self, component: str) -> float:
        """Retrieve unit price for specific electrical component.

        Queries database for exact component match and returns unit price.
        Returns 0.0 for unknown components to prevent estimation errors.
        No fuzzy matching or fallback pricing to maintain accuracy.

        Args:
            component (str): Exact component name (e.g., 'light_switch', 'outlet').
            Must match database component names exactly.

        Returns:
            float: Unit price in USD for the component. Returns 0.0 if component
            not found in database to avoid incorrect pricing.

        Algorithm:
            1. Execute SQL query for exact component name match
            2. Return first result's unit_price if found
            3. Return 0.0 if no matching component found
            4. Maintains data integrity by avoiding fallback pricing

        Related Functions:
            get_all_prices: Retrieves all component prices at once
            update_price: Modifies existing component prices

        """
        cursor = self.conn.execute('SELECT unit_price FROM prices WHERE component = ?', (component,))
        result = cursor.fetchone()
        return result[0] if result else 0.0
    
    def get_all_prices(self) -> dict:
        """Retrieve complete dictionary of all component prices.

        Fetches all electrical component prices from database as a dictionary
        for bulk operations, API responses, or bill generation. Useful for
        displaying price lists or bulk calculations.

        Args:
            None: Uses instance database connection.

        Returns:
            dict: Dictionary mapping component names (str) to unit prices (float).
            Example: {'light_switch': 18.75, 'outlet': 25.50, ...}

        Algorithm:
            1. Execute SQL query to fetch all component-price pairs
            2. Convert cursor results to dictionary using dict() constructor
            3. Return complete price mapping for all components

        Related Functions:
            get_price: Gets individual component price
            _init_db: Populates initial prices retrieved by this method

        """
        cursor = self.conn.execute('SELECT component, unit_price FROM prices')
        return dict(cursor.fetchall())
    
    def update_price(self, component: str, price: float):
        """Update or insert unit price for electrical component.

        Modifies existing component price or creates new component entry
        in database. Uses INSERT OR REPLACE for safe upsert operation.
        Commits transaction immediately for persistence.

        Args:
            component (str): Component name to update (e.g., 'light_switch').
            price (float): New unit price in USD. Must be non-negative value.

        Returns:
            None: Database update method with immediate commit.

        Algorithm:
            1. Execute INSERT OR REPLACE SQL with component name and new price
            2. Immediately commit transaction to persist changes
            3. Updates existing entries or creates new ones as needed

        Related Functions:
            get_price: Retrieves updated prices after modification
            _init_db: Sets initial prices that this method can later modify

        """
        self.conn.execute('INSERT OR REPLACE INTO prices VALUES (?, ?)', (component, price))
        self.conn.commit()
    
    def close(self):
        """Safely close database connection and release resources.

        Properly closes SQLite database connection to prevent resource leaks
        and ensure data integrity. Safe to call multiple times or when
        connection is already closed.

        Args:
            None: Uses instance database connection.

        Returns:
            None: Resource cleanup method.

        Algorithm:
            1. Check if database connection exists and is valid
            2. Close connection if available
            3. Safely handle cases where connection is already closed

        Related Functions:
            __init__: Creates the connection that this method closes
            Used in context managers and cleanup operations

        """
        if self.conn:
            self.conn.close() 