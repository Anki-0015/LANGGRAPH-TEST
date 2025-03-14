import { tool } from '@langchain/core/tools';
import { z } from "zod";
import { ChatOpenAI} from "@langchain/openai";
import { config } from 'dotenv';
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { SystemMessage, ToolMessage } from '@langchain/core/messages';

config();

const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
});

const multiply = tool(

    async(({a, b}) => {
        return a * b;
}), 
{
    name: 'multiply',
    description: 'Multiply two numbers',
    schema: z.object({
        a: z.number().describe('first number'),
        b: z.number().describe('second number'),
    }),
  }
);

const add = tool(
    async ({ a, b }) => {
      return a + b;
    },
    {
      name: "add",
      description: "Add two numbers together",
      schema: z.object({
        a: z.number().describe("first number"),
        b: z.number().describe("second number"),
      }),
    }
  );
  
  const divide = tool(
    async ({ a, b }) => {
      return a / b;
    },
    {
      name: "divide",
      description: "Divide two numbers",
      schema: z.object({
        a: z.number().describe("first number"),
        b: z.number().describe("second number"),
      }),
    }
  );

const tools = [add, multiply ,divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

async function llmCall(state) {
    // LLM decides whether to call a tool or not
    const result = await llmWithTools.invoke([
      {
        role: "system",
        content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
      },
      ...state.messages
    ]);
  
    return {
      messages: [result]
    };
}


async function toolNode(state) {
    // Performs the tool call
    const results = [];
    const lastMessage = state.messages.at(-1);

    if (lastMessage?.tool_calls?.length) {
        for (const toolCall of lastMessage.tool_calls) {
            const tool = toolsByName[toolCall.name];
            const observation = await tool.invoke(toolCall.args);
            results.push(
                new ToolMessage({
                    content: observation,
                    tool_call_id: toolCall.id,
                })
            );
        }
    }

    return { messages: results };
}


function shouldContinue(state) {
    const messages = state.messages;
    const lastMessage = messages.at(-1);

    // If the LLM makes a tool call, then perform an action
    if (lastMessage?.tool_calls?.length) {
        return "Action";
    }

    // Otherwise, we stop (reply to the user)
    return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
    .addNode('llmcall', llmCall)
    .addNode('tool', toolNode)

    .addEdge("__start__", "llmcall")

    .addConditionalEdges(
        "llmCall",
        shouldContinue,
        {
            // Name returned by shouldContinue : Name of next step
            "Action": "tools",
            "__end__": "__end__",
        }
    )
    .addEdge("tools", "llmcall")
    .compile();    

const messages = [{
    role: "user",
    content: "ADD 3 and 4",
},];

const result = await agentBuilder.invoke({ messages });
console.log(result.messages);