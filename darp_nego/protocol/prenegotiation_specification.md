# Mediator Prenegotiation Process Specification

## Overview

The prenegotiation phase in the Classic Single Mediated Text Mechanism establishes the foundation for the negotiation process by collecting client information from participating agents, analyzing client characteristics, and preparing the negotiation domain.

## Formal Specification

### Input Parameters

- **Agents**: Set of `BasicDARPNegotiator` instances participating in the negotiation
- **Road Network**: Optional `GISRoadNetwork` for routing calculations
- **Session Configuration**: Logging directory, session ID, and other configuration parameters

### Process Steps

#### Step 1: Domain Initialization
```
Initialize DARPNegotiationDomain()
Set clients_to_be_shared = []
Set client_revelations = []
```

#### Step 2: Initial State Logging
For each agent `a` in participants:
```
Log initial DARP solution for agent a
Store initial routing configuration
```

#### Step 3: Client Revelation Phase
For each agent `a` in participants:
```
clients_to_reveal = a.reveal_client()
if clients_to_reveal is not empty:
    for each client c in clients_to_reveal:
        clients_to_be_shared.append((a.agent_id, c))
        client_revelations.append((a.agent_id, c.client_id))
        domain.add_client(c, a.agent_id)
        Log: "Agent {a.agent_id} revealed client id: {c.client_id}"
```

#### Step 4: Information Distribution
For each (agent_id, client) in clients_to_be_shared:
```
For each agent a in participants:
    a.add_client_information(client, agent_id)
```

#### Step 5: Client Analysis

##### 5.1 Feature Vector Distance Analysis
```
client_vectors, distance_matrix, client_ids = domain._calculate_feature_vectors_and_distances()

if client_ids is not None:
    Print distance matrix between all client pairs
    Store analysis data for logging
```

**Feature Vector Components** (9-dimensional normalized vector):
- Pickup time window (late_pickup - early_pickup)
- Drop time window (late_drop - early_drop)  
- Total service time (late_drop - early_pickup)
- Travel distance (Euclidean distance between start and end coordinates)
- Client volume
- Pickup distance from depot (assumed at origin)
- Drop distance from depot
- Early pickup time
- Late drop time

##### 5.2 Client Cost Analysis
```
cost_analysis = domain.analyze_client_costs()

For each client c:
    Calculate average distance to all other clients
    Rank clients by cost (highest average distance = most costly)
    Categorize as: 🔴 most costly, 🟡 moderate, 🟢 least costly
```

#### Step 6: Logging and Finalization
```
logger.log_prenegotiation(
    participants, 
    domain, 
    distance_analysis, 
    cost_analysis, 
    client_revelations
)
```

### Output State

After prenegotiation, the mediator maintains:

1. **Domain State**: `DARPNegotiationDomain` containing:
   - `known_clients`: Dictionary mapping client_id → BasicDARPClient
   - `client_owners`: Dictionary mapping client_id → agent_id

2. **Client Pool**: `clients_to_be_shared` list of (agent_id, client) tuples

3. **Analysis Data**: 
   - Feature vector distance matrix
   - Client cost rankings
   - Revelation order tracking

4. **Agent Knowledge**: All agents informed about all revealed clients and their current owners

### Key Properties

#### Completeness
- All agents must reveal their willingness to share clients
- All revealed clients are added to the negotiation domain
- All agents receive complete information about all revealed clients

#### Transparency
- Client revelation order is tracked and logged
- Feature vector distances are calculated and displayed
- Client cost analysis is performed and shared

#### Consistency
- Domain state is consistent across all agents
- Client ownership is properly tracked
- All analysis is based on the complete set of revealed clients

### Error Handling

1. **No Clients Revealed**: If no agents reveal clients, negotiation cannot proceed
2. **Insufficient Clients**: If fewer than 2 clients are revealed, distance analysis is skipped
3. **Agent Communication**: If agent.reveal_client() fails, that agent is skipped

### Logging Requirements

The prenegotiation phase logs:
- Initial routing solutions for all agents
- Client revelation events and order
- Feature vector distance matrix
- Client cost analysis and rankings
- Complete domain state after revelation

### Transition to Negotiation

After prenegotiation completion:
- All agents have complete knowledge of the negotiation domain
- The mediator has all necessary information to generate outcomes
- The negotiation phase can begin with the first outcome proposal

## Mathematical Formulation

### Feature Vector Normalization
For client `c` with feature vector `f_c`:
```
f_normalized = (f_c - μ) / σ
```
Where `μ` and `σ` are the mean and standard deviation across all clients.

### Distance Calculation
Euclidean distance between clients `a` and `b`:
```
d(a,b) = ||f_a - f_b||_2
```

### Cost Ranking
Average distance for client `c`:
```
cost(c) = (1/(n-1)) * Σ_{i≠c} d(c,i)
```
Where `n` is the total number of clients.

This formal specification ensures that the prenegotiation phase establishes a complete, consistent, and analyzable foundation for the subsequent negotiation process. 