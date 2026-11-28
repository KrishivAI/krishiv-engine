// SPDX-License-Identifier: Apache-2.0
import { DropdownMenu as MenuPrimitive } from "bits-ui";

export { default as MenuContent } from "./menu-content.svelte";
export { default as MenuItem, menuItemVariants, type MenuItemVariant } from "./menu-item.svelte";
export { default as MenuLabel } from "./menu-label.svelte";
export { default as MenuSeparator } from "./menu-separator.svelte";

export const Menu = MenuPrimitive.Root;
export const MenuTrigger = MenuPrimitive.Trigger;
export const MenuGroup = MenuPrimitive.Group;
