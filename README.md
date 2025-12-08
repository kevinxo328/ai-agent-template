# AI Agent Template

## Local Development Setup

Follow these steps to run an agent locally:

### 1. Navigate to the Agent Directory

Go to the `agents` folder and select the agent you want to run, for example:

```bash
cd agents/demo
```

### 2. Configure Environment Variables

- Rename `.env.example` to `.env`:
  ```bash
  mv .env.example .env
  ```
- Open the `.env` file and fill in the required variables

### 3. Start the Agent

Run the following command to start the agent in development mode:

```bash
langgraph dev --config ./agents/demo/langgraph.json
```

Replace `./agents/demo/langgraph.json` with the path to your agent's configuration file if it's different.
