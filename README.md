# NovaMart — Enterprise Multi-Agent Customer Support System

**Udacity | Future AWS Agentic Engineer Nanodegree | Course 3, Project 3**

## Project overview

NovaMart is a multi-agent customer-support application built with the **Strands Agents SDK** and **Amazon Bedrock AgentCore**. An orchestrator routes customer requests to specialized agents for order lookup, policy retrieval, return initiation, and customer communication. The system uses shared workflow state, Amazon Bedrock Knowledge Bases, guardrails, memory, and AWS observability.

This project was implemented using the provided Udacity Project 3 starter code and infrastructure. The original starter instructions are retained separately in [`README_starter.md`](README_starter.md).

## Architecture

```text
Customer request
       |
OrchestratorAgent (Claude Haiku 4.5)
       |
       +-- InventoryAgent (Claude Sonnet 4.5)
       |       +-- Order/customer data in DynamoDB
       |
       +-- PolicyAgent (Claude Sonnet 4.5)
       |       +-- ReturnsPolicyRetriever  -> Returns Knowledge Base
       |       +-- ShippingPolicyRetriever -> Shipping Knowledge Base
       |       +-- WarrantyPolicyRetriever -> Warranty Knowledge Base
       |           (retrievers run in parallel)
       |
       +-- RefundAgent (Claude Sonnet 4.5)
       |       +-- Return eligibility and initiation
       |
       +-- CommunicationAgent (Claude Sonnet 4.5)
               +-- Final customer-facing answer

Shared context: DynamoDB WorkflowStateTable
Observability: CloudWatch Logs and AWS X-Ray
```

The application follows these routing rules:

- **Returns/refunds:** Orchestrator → InventoryAgent → RefundAgent → CommunicationAgent
- **Policy questions:** Orchestrator → PolicyAgent (three parallel retrievers) → CommunicationAgent
- **Calculations:** Orchestrator → CommunicationAgent
- **Every request:** initialize the workflow session first, and delegate the final response to CommunicationAgent.

## Implemented features

- Five Strands agents: Orchestrator, Inventory, Policy, Refund, and Communication.
- Three specialized policy retriever sub-agents using Amazon Bedrock Knowledge Bases for returns, shipping, and warranty documents, executed in parallel with `ThreadPoolExecutor`.
- DynamoDB-backed customer/order lookups and shared workflow state with optimistic locking.
- Return eligibility logic with Standard (30-day) and Premium (60-day) return windows.
- Amazon Bedrock Guardrails, including content safeguards and sensitive-information handling.
- Amazon Bedrock AgentCore Runtime deployment and AgentCore Memory with a session-summary strategy and seven-day event retention.
- CloudWatch logging and X-Ray tracing for end-to-end workflow observability.

**Implementation note:** The optional AgentCore Gateway/MCP extension was not included; it is not required for the Project 3 grading rubric.

## Technology stack

Python 3.12, Strands Agents SDK, Amazon Bedrock (Claude Haiku 4.5 / Claude Sonnet 4.5), Amazon Bedrock AgentCore, Bedrock Knowledge Bases, S3 Vectors, DynamoDB, CloudFormation, CloudWatch, and AWS X-Ray. Deployment region: `us-east-1`.

## Verification and test results

The full automated suite passed with **120/120 points**:

```bash
python tests/test_agent.py all
```

Three end-to-end requests were also executed against the deployed AgentCore Runtime:

| Scenario | Observed result | Verification |
|---|---|---|
| Premium return-policy question | Returned the 60-day Premium window | PolicyAgent and all three KB retriever spans observed in X-Ray |
| 5 items at $29.99 with 10% off | Correct final total: **$134.96** | CommunicationAgent routing checked in X-Ray |
| Return order `ORD-27176` for `CUST-001` | Initiated return `RET-EB442DA80C` | DynamoDB order status changed from `delivered` to `return_initiated`; Inventory → Refund → Communication routing checked in X-Ray |

**Note:** The return test initiated a return; it did **not** demonstrate that a payment refund had been disbursed.

## Screenshots and evidence

### Required evidence

**1. Automated Test Suite — 120/120 Points**

All automated tests passed successfully, achieving a final score of 120/120 (100%).

The complete test output is documented across six screenshots:

- [Test Results — Part 1](submission/screenshots/120of120-AllTestsPassed-p1.png)
- [Test Results — Part 2](submission/screenshots/120of120-AllTestsPassed-p2.png)
- [Test Results — Part 3](submission/screenshots/120of120-AllTestsPassed-p3.png)
- [Test Results — Part 4](submission/screenshots/120of120-AllTestsPassed-p4.png)
- [Test Results — Part 5](submission/screenshots/120of120-AllTestsPassed-p5.png)

**Final Test Score:**

![Automated Test Suite — 120/120](submission/screenshots/120of120-AllTestsPassed-p6.png)

**2. AWS X-Ray Service Map**

[View X-Ray Service Map](submission/screenshots/xray_service_map.png)

### Additional end-to-end evidence

3. [Premium return-policy response](submission/screenshots/premium_return_policy.png)
4. [Discount calculation response](submission/screenshots/discount_calculation.png)
5. [Return initiation response](submission/screenshots/return_initiated.png)
6. [DynamoDB return status updated](submission/screenshots/dynamodb_return_update.png)

## Running the project

> Requires access to the relevant AWS resources and permissions in `us-east-1`. The deployment described here used the temporary AWS credentials supplied for the Udacity lab; no credentials are included in the submission.

Install dependencies and configure the environment:

```bash
pip install -r requirements.txt
cp .env.example .env
python config.py
```

Populate the `.env` values for the three Knowledge Base IDs, Runtime ARN, and Guardrail ID/version as described in [`README_starter.md`](README_starter.md). The original starter README contains the complete CloudFormation, Knowledge Base creation, and deployment instructions.

To run the automated verification suite:

```bash
python tests/test_agent.py all
```

To invoke the deployed Runtime with a read-only policy query:

```bash
python src/agent_orchestrator.py invoke \
  'What is the return policy for premium customers?' \
  CUST-001
```

**Do not rerun the same return-initiation request without inspecting the existing order state**, as it may change the seeded DynamoDB data.

## Main project files

- `src/agent_orchestrator.py` — implemented agent builders, tools, routing, deployment, memory, and observability configuration.
- `src/bedrock_kb_retrieval.py` — provided Knowledge Base retrieval helper.
- `src/agent_observability.py` — provided X-Ray and CloudWatch instrumentation.
- `config.py` — provided AWS configuration loader.
- `infrastructure/` — supplied infrastructure, data seeding, and cleanup scripts.
- `tests/test_agent.py` — supplied automated grading tests.
- `submission/screenshots/` — submission evidence.

## Outcome

The project demonstrates a deployed, observable multi-agent support workflow with specialist agents, parallel policy retrieval, DynamoDB-backed state changes, and guardrail/memory integration. Automated tests passed **120/120**, and the three required customer scenarios were exercised successfully against the deployed application.

**Security:** Real AWS access keys, session tokens, and the populated `.env` file are intentionally excluded from the submission.
