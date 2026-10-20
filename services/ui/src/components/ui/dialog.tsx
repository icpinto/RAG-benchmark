import * as Dialog from "@radix-ui/react-dialog";
import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { X, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "./button";
export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="overlay" />
        <Dialog.Content className="modal">
          <Dialog.Title>{title}</Dialog.Title>
          <Dialog.Description>
            {description || "Review and update the fields below."}
          </Dialog.Description>
          <Dialog.Close asChild>
            <Button
              variant="ghost"
              size="icon"
              className="modal-close"
              aria-label="Close dialog"
            >
              <X aria-hidden="true" focusable="false" size={18} />
            </Button>
          </Dialog.Close>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function Confirm({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  confirmIcon: ConfirmIcon,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  confirmIcon?: LucideIcon;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <AlertDialog.Root
      open={open}
      onOpenChange={(value) => {
        if (!value) onCancel();
      }}
    >
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="overlay" />
        <AlertDialog.Content className="modal modal-small">
          <AlertDialog.Title>{title}</AlertDialog.Title>
          <AlertDialog.Description>{description}</AlertDialog.Description>
          <div className="actions justify-end">
            <AlertDialog.Cancel asChild>
              <Button variant="outline" onClick={onCancel}>
                <X aria-hidden="true" focusable="false" size={16} />
                Cancel
              </Button>
            </AlertDialog.Cancel>
            <AlertDialog.Action asChild>
              <Button onClick={onConfirm}>
                {ConfirmIcon && (
                  <ConfirmIcon aria-hidden="true" focusable="false" size={16} />
                )}
                {confirmLabel}
              </Button>
            </AlertDialog.Action>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
