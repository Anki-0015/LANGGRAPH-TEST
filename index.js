import { DynamicTool } from '@langchain/core/tools';
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { config } from 'dotenv';
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { SystemMessage, ToolMessage } from '@langchain/core/messages';

config();

const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
});

const multiply = new DynamicTool({
    name: 'multiply',
    description: 'Multiply two numbers',
    func: async ({ a, b }) => a * b,
    schema: z.object({
        a: z.number().describe('first number'),
        b: z.number().describe('second number'),
    }),
});

const add = new DynamicTool({
    name: "add",
    description: "Add two numbers together",
    func: async ({ a, b }) => a + b,
    schema: z.object({
        a: z.number().describe("first number"),
        b: z.number().describe("second number"),
    }),
});

const divide = new DynamicTool({
    name: "divide",
    description: "Divide two numbers",
    func: async ({ a, b }) => {
        if (b === 0) {
            throw new Error("Division by zero is not allowed.");
        }
        return a / b;
    },
    schema: z.object({
        a: z.number().describe("first number"),
        b: z.number().describe("second number"),
    }),
});

const tools = [add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

async function llmCall(state) {
    const result = await llmWithTools.invoke([
        {
            role: "system",
            content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        },
        ...state.messages
    ]);

    return { messages: [result] };
}

async function toolNode(state) {
    const results = [];
    const lastMessage = state.messages.at(-1);

    if (lastMessage?.tool_calls?.length) {
        for (const toolCall of lastMessage.tool_calls) {
            const tool = toolsByName[toolCall.name];
            if (tool) {
                const observation = await tool.invoke(toolCall.args);
                results.push(
                    new ToolMessage({
                        content: observation.toString(),
                        tool_call_id: toolCall.id,
                    })
                );
            }
        }
    }

    return { messages: results };
}

function shouldContinue(state) {
    const lastMessage = state.messages.at(-1);

    if (lastMessage?.tool_calls?.length) {
        return "Action";
    }
    return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
    .addNode('llmcall', llmCall)
    .addNode('tools', toolNode)
    .addEdge("__start__", "llmcall")
    .addConditionalEdges(
        "llmcall",
        shouldContinue,
        {
            "Action": "tools",
            "__end__": "__end__",
        }
    )
    .addEdge("tools", "llmcall")
    .compile();

(async () => {
    const messages = [{
        role: "user",
        content: "ADD 3 and 4",
    }];

    const result = await agentBuilder.invoke({ messages });
    console.log(result.messages);
})();
