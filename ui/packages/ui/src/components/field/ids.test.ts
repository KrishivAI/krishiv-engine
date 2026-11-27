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
  test("marks the control invalid only with an error", () => {
    expect(fieldControl("f", true, false)).toEqual({ id: "f", "aria-describedby": "f-help", "aria-invalid": undefined });
    expect(fieldControl("f", false, true)["aria-invalid"]).toBe("true");
  });
});
