import React, { useEffect, useState } from "react";
import { Input } from "../components/ui/Input";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";

const STORAGE_KEY = "agropredict_tasks";

const Todo = () => {
  const [taskText, setTaskText] = useState("");
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setTasks(JSON.parse(stored));
      } catch {
        setTasks([]);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
  }, [tasks]);

  const addTask = (e) => {
    e.preventDefault();
    if (!taskText.trim()) return;
    const newTask = {
      id: Date.now(),
      text: taskText.trim(),
      done: false
    };
    setTasks([newTask, ...tasks]);
    setTaskText("");
  };

  const toggleTask = (id) => {
    setTasks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t))
    );
  };

  const deleteTask = (id) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  };

  const hasTasks = tasks.length > 0;

  return (
    <div className="pt-3 pb-8">
      <p className="text-xs text-gray-500 mb-3">
        Plan your farm work for the day: irrigation, spraying, harvesting and
        more.
      </p>

      <form onSubmit={addTask} className="mb-3">
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <Input
              label="New task"
              placeholder="e.g. Check water level in paddy field"
              value={taskText}
              onChange={(e) => setTaskText(e.target.value)}
            />
          </div>
          <div className="hidden sm:block">
            <Button type="submit" className="px-4 py-3 text-sm">
              Add
            </Button>
          </div>
        </div>
        <div className="sm:hidden mt-1">
          <Button type="submit">Add Task</Button>
        </div>
      </form>

      {!hasTasks && (
        <Card className="border border-dashed border-gray-300 bg-gray-50">
          <p className="text-sm font-medium text-gray-700 mb-1">
            No tasks yet
          </p>
          <p className="text-[11px] text-gray-500">
            Add today&apos;s important jobs: weeding, fertilizer application,
            repair pump, visit market, etc.
          </p>
        </Card>
      )}

      <div className="space-y-2 mt-2">
        {tasks.map((task) => (
          <Card
            key={task.id}
            className={`flex items-center gap-3 ${
              task.done ? "bg-agro-greenSoft/70" : "bg-white"
            }`}
          >
            <button
              onClick={() => toggleTask(task.id)}
              className={`w-7 h-7 rounded-full border flex items-center justify-center text-lg active:scale-95 transition ${
                task.done
                  ? "bg-agro-green text-white border-agro-green"
                  : "border-gray-300 text-gray-400"
              }`}
              aria-label={task.done ? "Mark as pending" : "Mark as completed"}
            >
              {task.done ? "✓" : ""}
            </button>
            <div className="flex-1">
              <p
                className={`text-sm ${
                  task.done
                    ? "line-through text-gray-500"
                    : "text-agro-dark font-medium"
                }`}
              >
                {task.text}
              </p>
              <p className="text-[10px] text-gray-400 mt-0.5">
                Tap the circle to mark done. Long-press delete to remove.
              </p>
            </div>
            <button
              onClick={() => deleteTask(task.id)}
              className="text-xs text-red-500 px-2 py-1 rounded-full bg-red-50 hover:bg-red-100 active:scale-95"
            >
              Delete
            </button>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default Todo;


