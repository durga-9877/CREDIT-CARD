import hashlib
import json
import time
from typing import List, Dict, Any
from utils.logging_config import logger

class Block:
    def __init__(self, index: int, transactions: List[Dict], timestamp: float, previous_hash: str, proof: int):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.proof = proof
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "proof": self.proof
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict] = []
        self.difficulty = 4  # Adjustable difficulty for proof of work
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], time.time(), "0", 0)
        self.chain.append(genesis_block)
        logger.info("Genesis block created")

    def add_transaction(self, transaction_id: str, prediction: str, amount: float) -> int:
        transaction = {
            'transaction_id': transaction_id,
            'prediction': prediction,
            'amount': amount,
            'timestamp': time.time()
        }
        self.current_transactions.append(transaction)
        logger.info(f"Transaction added: {transaction_id}")
        return self.last_block.index + 1

    def proof_of_work(self, last_proof: int) -> int:
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    def valid_proof(self, last_proof: int, proof: int) -> bool:
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:self.difficulty] == '0' * self.difficulty

    def mine_block(self) -> Block:
        if not self.current_transactions:
            raise ValueError("No transactions to mine")

        last_block = self.last_block
        last_proof = last_block.proof
        proof = self.proof_of_work(last_proof)

        block = Block(
            index=len(self.chain),
            transactions=self.current_transactions,
            timestamp=time.time(),
            previous_hash=last_block.hash,
            proof=proof
        )

        self.current_transactions = []
        self.chain.append(block)
        logger.info(f"Block mined: {block.hash}")
        return block

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def validate_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.compute_hash():
                logger.error(f"Invalid hash in block {current_block.index}")
                return False

            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash in block {current_block.index}")
                return False

            if not self.valid_proof(previous_block.proof, current_block.proof):
                logger.error(f"Invalid proof of work in block {current_block.index}")
                return False

        logger.info("Blockchain validation successful")
        return True

    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        for block in self.chain:
            for transaction in block.transactions:
                if transaction['transaction_id'] == transaction_id:
                    return transaction
        return None 