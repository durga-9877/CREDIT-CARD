import hashlib
import json
import time
from typing import Dict, List, Optional

class Block:
    def __init__(self, index: int, prediction: str, amount: float, time: float, previous_hash: str):
        self.index = index
        self.prediction = prediction
        self.amount = amount
        self.time = time
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = f"{self.index}{self.prediction}{self.amount}{self.time}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_block = Block(
            index=0,
            prediction="Genesis",
            amount=0,
            time=0,
            previous_hash='0' * 64
        )
        self.chain.append(genesis_block)

    def add_block(self, prediction: str, amount: float, time: float) -> Block:
        """Add a new block to the chain."""
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            prediction=prediction,
            amount=amount,
            time=time,
            previous_hash=previous_block.hash
        )
        self.chain.append(new_block)
        return new_block

    def get_chain(self) -> List[Dict]:
        """Get the entire blockchain."""
        return [{
            'block_index': block.index,
            'prediction': block.prediction,
            'amount': block.amount,
            'time': block.time,
            'block_hash': block.hash,
            'previous_hash': block.previous_hash
        } for block in self.chain]

    def get_transaction_by_amount(self, amount: float) -> Optional[Dict]:
        """Get a transaction by its amount."""
        for block in self.chain:
            if abs(block.amount - amount) < 0.01:  # Allow for floating point comparison
                return {
                    'block_index': block.index,
                    'prediction': block.prediction,
                    'amount': block.amount,
                    'time': block.time,
                    'block_hash': block.hash,
                    'previous_hash': block.previous_hash
                }
        return None

    def validate_chain(self) -> bool:
        """Validate the entire blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the hash is correct
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if the previous hash is correct
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True 