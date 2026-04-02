import { redirect } from "next/navigation";

export default function CalibrationRedirect() {
  redirect("/annotate/calibrate");
}
