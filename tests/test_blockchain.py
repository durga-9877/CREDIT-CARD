import pytest
import time
from app.blockchain.blockchain import Blockchain, Block

@pytest.fixture
def blockchain():
    return Blockchain()

def test_create_genesis_block(blockchain):
    assert len(blockchain.chain) == 1
    assert blockchain.chain[0].index == 0
    assert blockchain.chain[0].previous_hash == "0"

def test_add_transaction(blockchain):
    transaction_id = "test123"
    prediction = "Legitimate"
    amount = 100.0
    
    index = blockchain.add_transaction(transaction_id, prediction, amount)
    assert index == 1
    assert len(blockchain.current_transactions) == 1
    assert blockchain.current_transactions[0]['transaction_id'] == transaction_id

def test_mine_block(blockchain):
    # Add a transaction first
    blockchain.add_transaction("test123", "Legitimate", 100.0)
    
    # Mine the block
    block = blockchain.mine_block()
    
    assert isinstance(block, Block)
    assert len(blockchain.chain) == 2
    assert len(blockchain.current_transactions) == 0

def test_proof_of_work(blockchain):
    last_proof = 123
    proof = blockchain.proof_of_work(last_proof)
    
    assert blockchain.valid_proof(last_proof, proof)
    assert not blockchain.valid_proof(last_proof, proof - 1)

def test_validate_chain(blockchain):
    # Add some transactions and mine blocks
    blockchain.add_transaction("test1", "Legitimate", 100.0)
    blockchain.mine_block()
    blockchain.add_transaction("test2", "Fraudulent", 200.0)
    blockchain.mine_block()
    
    # Validate the chain
    assert blockchain.validate_chain()

def test_validate_chain_tampering(blockchain):
    # Add some transactions and mine blocks
    blockchain.add_transaction("test1", "Legitimate", 100.0)
    blockchain.mine_block()
    
    # Tamper with the chain
    blockchain.chain[1].transactions[0]['amount'] = 999.0
    
    # Validate the chain
    assert not blockchain.validate_chain()

def test_get_transaction(blockchain):
    # Add a transaction and mine the block
    transaction_id = "test123"
    blockchain.add_transaction(transaction_id, "Legitimate", 100.0)
    blockchain.mine_block()
    
    # Get the transaction
    transaction = blockchain.get_transaction(transaction_id)
    
    assert transaction is not None
    assert transaction['transaction_id'] == transaction_id
    assert transaction['amount'] == 100.0 