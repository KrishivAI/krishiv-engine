<script lang="ts">
  import { Dialog as DialogPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { class: className, ...rest }: DialogPrimitive.TitleProps = $props();
</script>

<DialogPrimitive.Title level={2} class={cn("px-5 pt-4.5 pb-1.5 font-display text-lg leading-6 font-bold", className)} {...rest} />
