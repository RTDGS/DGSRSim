const scrollButton = document.querySelector(".scroll-top");

function updateScrollButton() {
  if (!scrollButton) return;
  scrollButton.classList.toggle("visible", window.scrollY > 500);
}

window.addEventListener("scroll", updateScrollButton, { passive: true });
updateScrollButton();

if (scrollButton) {
  scrollButton.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    const text = target.innerText.trim();
    await copyText(button, text);
  });
});

document.querySelectorAll("[data-copy-value]").forEach((button) => {
  button.addEventListener("click", async () => {
    await copyText(button, button.dataset.copyValue || "");
  });
});

async function copyText(button, text) {
  if (!text) return;
  const previous = button.textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    button.textContent = "Copied";
  } catch {
    button.textContent = "Copy failed";
  } finally {
    window.setTimeout(() => {
      button.textContent = previous;
    }, 1400);
  }
}
