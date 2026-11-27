// SPDX-License-Identifier: Apache-2.0
import { Dialog as DialogPrimitive } from "bits-ui";

export { default as DialogContent } from "./dialog-content.svelte";
export { default as DialogTitle } from "./dialog-title.svelte";
export { default as DialogDescription } from "./dialog-description.svelte";
export { default as DialogFooter } from "./dialog-footer.svelte";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;
