import { redirect } from "next/navigation";

export default function AnnotateOverlayRedirect() {
  redirect("/evaluate/overlay");
}
