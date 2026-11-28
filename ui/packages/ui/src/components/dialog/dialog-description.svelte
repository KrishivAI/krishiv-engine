<script lang="ts">
  import { Dialog as DialogPrimitive } from "bits-ui";
  import { cn } from "../../utils";

  let { class: className, ...rest }: DialogPrimitive.DescriptionProps = $props();
</script>

<DialogPrimitive.Description class={cn("px-5 pt-1.5 pb-4 text-text-2", className)} {...rest} />
