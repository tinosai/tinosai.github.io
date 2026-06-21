// Small terminal flourishes — progressive enhancement only.
(function () {
  // Live-ish clock in the top bar title.
  var clock = document.querySelector("[data-clock]");
  if (clock) {
    var tick = function () {
      var d = new Date();
      var p = function (n) { return String(n).padStart(2, "0"); };
      clock.textContent = p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
    };
    tick();
    setInterval(tick, 1000);
  }

  // Typewriter effect for the hero command (respects reduced-motion).
  var typed = document.querySelector("[data-type]");
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (typed && !reduce) {
    var full = typed.getAttribute("data-type");
    typed.textContent = "";
    var i = 0;
    var step = function () {
      if (i <= full.length) {
        typed.textContent = full.slice(0, i);
        i++;
        setTimeout(step, 45);
      }
    };
    step();
  } else if (typed) {
    typed.textContent = typed.getAttribute("data-type");
  }
})();
