// SPDX-License-Identifier: Apache-2.0
import { mount } from "svelte";
import "./app.css";
import App from "./App.svelte";

const target = document.getElementById("app");
if (!target) throw new Error("missing #app");
export default mount(App, { target });
