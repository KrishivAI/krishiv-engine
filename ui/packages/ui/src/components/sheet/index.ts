import { Dialog as SheetPrimitive } from "bits-ui";

export { default as SheetContent } from "./sheet-content.svelte";

export const Sheet = SheetPrimitive.Root;
export const SheetTrigger = SheetPrimitive.Trigger;
export const SheetClose = SheetPrimitive.Close;
export const SheetTitle = SheetPrimitive.Title;
export const SheetDescription = SheetPrimitive.Description;
