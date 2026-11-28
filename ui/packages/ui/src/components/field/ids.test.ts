// SPDX-License-Identifier: Apache-2.0
import { describe, expect, test } from "vitest";
import { fieldControl, fieldIds } from "./ids";

describe("fieldIds", () => {
  test("names nothing without help or error", () => {
    expect(fieldIds("f", false, false).describedby).toBeUndefined();
  });
  test("names the error before the help", () => {
    expect(fieldIds("f", true, true).describedby).toBe("f-error f-help");
    expect(fieldIds("f", true, false).describedby).toBe("f-help");
  });
  test("keeps the caller's description ids after the field's, once each", () => {
    expect(fieldIds("f", true, true, "external-help").describedby).toBe("f-error f-help external-help");
    expect(fieldIds("f", false, false, " a  b a ").describedby).toBe("a b");
    expect(fieldIds("f", true, false, "f-help").describedby).toBe("f-help");
    expect(fieldControl("f", false, true, { describedby: "external-help" })["aria-describedby"]).toBe("f-error external-help");
  });
  test("marks the control invalid with an error, whatever the caller says", () => {
    expect(fieldControl("f", true, false)).toEqual({ id: "f", "aria-describedby": "f-help", "aria-invalid": undefined });
    expect(fieldControl("f", false, true)["aria-invalid"]).toBe("true");
    expect(fieldControl("f", false, true, { invalid: false })["aria-invalid"]).toBe("true");
    expect(fieldControl("f", false, true, { invalid: "false" })["aria-invalid"]).toBe("true");
  });
  test("keeps the caller's own invalid state without an error", () => {
    expect(fieldControl("f", false, false, { invalid: true })["aria-invalid"]).toBe("true");
    expect(fieldControl("f", false, false, { invalid: "true" })["aria-invalid"]).toBe("true");
    expect(fieldControl("f", false, false, { invalid: false })["aria-invalid"]).toBeUndefined();
  });
});
